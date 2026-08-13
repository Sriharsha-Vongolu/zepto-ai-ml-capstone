from pathlib import Path
from urllib.parse import urljoin
import re
import sqlite3

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

MODULE_DIR = Path(__file__).resolve().parent
DB_PATH = MODULE_DIR / "books.db"
SQL_OUTPUT_PATH = MODULE_DIR / "sql_query_outputs.txt"
PANDAS_OUTPUT_PATH = MODULE_DIR / "pandas_join_comparison.txt"

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


QUERIES = {
    "01_select_where": """
        SELECT title, rating, in_stock
        FROM books
        WHERE rating = 5
          AND in_stock = 1
        ORDER BY title;
    """,

    "02_order_by_limit": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "03_distinct": """
        SELECT DISTINCT category_name AS category
        FROM categories
        ORDER BY category;
    """,

    "04_between": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,

    "05_join": """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name AS category
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name ASC,
            b.rating DESC,
            b.book_id ASC;
    """,
}


def get_soup(session, url):
    """Download one page and return a BeautifulSoup object."""

    response = session.get(url, timeout=30)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_category_links(session):
    """Get all individual book-category links from the home page."""

    soup = get_soup(session, BASE_URL)

    categories = []

    for link in soup.select(".side_categories ul li ul li a"):
        category_name = link.get_text(strip=True)
        category_url = urljoin(BASE_URL, link["href"])

        categories.append((category_name, category_url))

    if not categories:
        raise RuntimeError("No categories were found on the website.")

    return categories


def scrape_category(session, category_name, category_url):
    """Scrape every paginated listing page for one category."""

    books = []
    current_url = category_url

    while current_url:
        soup = get_soup(session, current_url)

        products = soup.select("article.product_pod")

        for product in products:
            title_element = product.select_one("h3 a")
            price_element = product.select_one(".price_color")
            rating_element = product.select_one("p.star-rating")
            availability_element = product.select_one(".availability")

            title = (
                title_element.get("title")
                if title_element
                else None
            )

            price = (
                price_element.get_text(strip=True)
                if price_element
                else None
            )

            star_rating = None

            if rating_element:
                rating_classes = rating_element.get("class", [])

                star_rating = next(
                    (
                        class_name
                        for class_name in rating_classes
                        if class_name != "star-rating"
                    ),
                    None,
                )

            availability = (
                " ".join(availability_element.stripped_strings)
                if availability_element
                else None
            )

            books.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category_name,
                }
            )

        next_link = soup.select_one("li.next a")

        if next_link:
            current_url = urljoin(current_url, next_link["href"])
        else:
            current_url = None

    return books


def parse_price(value):
    """Convert a displayed price into a float."""

    if not isinstance(value, str):
        return None

    match = re.search(r"\d+(?:\.\d+)?", value)

    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def parse_rating(value):
    """Convert One-Five text ratings into integers."""

    return RATING_MAP.get(value)


def parse_availability(value):
    """Convert availability text into a boolean value."""

    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()

    if "out of stock" in normalized:
        return False

    if "in stock" in normalized:
        return True

    return None


def clean_data(raw_df):
    """Clean and convert all scraped fields."""

    df = raw_df.copy()

    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_availability)

    invalid_mask = (
        df["title"].fillna("").astype(str).str.strip().eq("")
        | df["category"].fillna("").astype(str).str.strip().eq("")
        | df["price_gbp"].isna()
        | df["rating"].isna()
        | df["in_stock"].isna()
    )

    dropped_df = df[invalid_mask].copy()
    clean_df = df[~invalid_mask].copy()

    clean_df["price_gbp"] = clean_df["price_gbp"].astype(float)
    clean_df["rating"] = clean_df["rating"].astype(int)
    clean_df["in_stock"] = clean_df["in_stock"].astype(bool)

    clean_df["price_inr"] = (
        clean_df["price_gbp"] * GBP_TO_INR
    )

    clean_df = clean_df[
        [
            "title",
            "price",
            "star_rating",
            "availability",
            "category",
            "price_gbp",
            "rating",
            "in_stock",
            "price_inr",
        ]
    ]

    return clean_df, dropped_df


def scrape_required_scope(session):
    """
    Scrape complete categories until the cleaned data contains
    at least 60 rows across at least 3 categories.
    """

    category_links = get_category_links(session)

    all_rows = []

    for category_name, category_url in category_links:
        print(f"Scraping category: {category_name}")

        category_rows = scrape_category(
            session=session,
            category_name=category_name,
            category_url=category_url,
        )

        all_rows.extend(category_rows)

        current_raw_df = pd.DataFrame(all_rows)
        current_clean_df, _ = clean_data(current_raw_df)

        clean_count = len(current_clean_df)
        category_count = current_clean_df["category"].nunique()

        print(
            f"Current cleaned rows: {clean_count}, "
            f"categories: {category_count}"
        )

        if clean_count >= 60 and category_count >= 3:
            return current_raw_df

    raise RuntimeError(
        "Unable to obtain at least 60 cleaned books "
        "across at least 3 categories."
    )


def create_normalized_dataframes(clean_df):
    """Create in-memory normalized categories and books DataFrames."""

    category_names = sorted(
        clean_df["category"].drop_duplicates().tolist()
    )

    categories_df = pd.DataFrame(
        {
            "category_id": range(1, len(category_names) + 1),
            "category_name": category_names,
        }
    )

    category_map = dict(
        zip(
            categories_df["category_name"],
            categories_df["category_id"],
        )
    )

    books_df = clean_df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ].copy()

    books_df["category_id"] = (
        books_df["category"]
        .map(category_map)
        .astype(int)
    )

    books_df = books_df.drop(columns=["category"])

    books_df.insert(
        0,
        "book_id",
        range(1, len(books_df) + 1),
    )

    return categories_df, books_df


def create_database(categories_df, books_df):
    """Create the normalized SQLite database from scratch."""

    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA foreign_keys = ON;")

    connection.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL
                CHECK (rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL
                CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        );
        """
    )

    category_records = [
        (
            int(row.category_id),
            row.category_name,
        )
        for row in categories_df.itertuples(index=False)
    ]

    connection.executemany(
        """
        INSERT INTO categories (
            category_id,
            category_name
        )
        VALUES (?, ?);
        """,
        category_records,
    )

    book_records = [
        (
            int(row.book_id),
            row.title,
            float(row.price_gbp),
            float(row.price_inr),
            int(row.rating),
            int(row.in_stock),
            int(row.category_id),
        )
        for row in books_df.itertuples(index=False)
    ]

    connection.executemany(
        """
        INSERT INTO books (
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        book_records,
    )

    connection.commit()

    return connection


def execute_and_save_queries(connection):
    """Execute all required SQL queries and save query + output."""

    output_sections = []

    for query_name, query in QUERIES.items():
        cursor = connection.execute(query)

        columns = [
            description[0]
            for description in cursor.description
        ]

        rows = cursor.fetchall()

        result_df = pd.DataFrame(
            rows,
            columns=columns,
        )

        section = (
            f"{query_name}\n"
            f"{'=' * len(query_name)}\n\n"
            f"{query.strip()}\n\n"
            f"{result_df.to_string(index=False)}"
        )

        output_sections.append(section)

        print()
        print(section)
        print()

    SQL_OUTPUT_PATH.write_text(
        "\n\n\n".join(output_sections),
        encoding="utf-8",
    )


def validate_with_pandas(
    connection,
    books_df,
    categories_df,
):
    """
    Read two SQL queries with pd.read_sql and reproduce
    the JOIN using pd.merge.
    """

    order_limit_df = pd.read_sql(
        QUERIES["02_order_by_limit"],
        connection,
    )

    sql_join_df = pd.read_sql(
        QUERIES["05_join"],
        connection,
    )

    pandas_join_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner",
    )

    pandas_join_df = pandas_join_df.rename(
        columns={
            "category_name": "category",
        }
    )

    # SQLite stores booleans as INTEGER values.
    pandas_join_df["in_stock"] = (
        pandas_join_df["in_stock"].astype(int)
    )

    pandas_join_df = pandas_join_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ]

    pandas_join_df = (
        pandas_join_df
        .sort_values(
            by=[
                "category",
                "rating",
                "book_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    sql_join_df = sql_join_df.reset_index(drop=True)

    equivalent = sql_join_df.equals(pandas_join_df)

    side_by_side_df = pd.concat(
        [
            sql_join_df.add_prefix("sql_"),
            pandas_join_df.add_prefix("pandas_"),
        ],
        axis=1,
    )

    output_text = (
        "pd.read_sql result — ORDER BY / LIMIT query\n"
        "============================================\n\n"
        f"{order_limit_df.to_string(index=False)}\n\n\n"
        "pd.read_sql result — JOIN query\n"
        "===============================\n\n"
        f"{sql_join_df.to_string(index=False)}\n\n\n"
        "pd.merge reproduction of JOIN\n"
        "=============================\n\n"
        f"{pandas_join_df.to_string(index=False)}\n\n\n"
        "SQL JOIN and pandas merge side-by-side\n"
        "======================================\n\n"
        f"{side_by_side_df.to_string(index=False)}\n\n"
        f"Equivalent outputs: {equivalent}\n"
    )

    PANDAS_OUTPUT_PATH.write_text(
        output_text,
        encoding="utf-8",
    )

    print()
    print("Pandas JOIN validation")
    print("----------------------")
    print(f"Equivalent outputs: {equivalent}")

    if not equivalent:
        raise AssertionError(
            "SQL JOIN and pandas merge outputs do not match."
        )


def main():
    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent":
                    "AI-ML-Capstone-Data-Pipeline/1.0"
            }
        )

        raw_df = scrape_required_scope(session)

    clean_df, dropped_df = clean_data(raw_df)

    if len(clean_df) < 60:
        raise AssertionError(
            "Cleaned dataset contains fewer than 60 books."
        )

    if clean_df["category"].nunique() < 3:
        raise AssertionError(
            "Cleaned dataset contains fewer than 3 categories."
        )

    categories_df, books_df = (
        create_normalized_dataframes(clean_df)
    )

    connection = create_database(
        categories_df,
        books_df,
    )

    try:
        execute_and_save_queries(connection)

        validate_with_pandas(
            connection,
            books_df,
            categories_df,
        )

    finally:
        connection.close()

    print()
    print("Pipeline completed successfully")
    print("-------------------------------")
    print(f"Raw rows scraped: {len(raw_df)}")
    print(f"Rows dropped during cleaning: {len(dropped_df)}")
    print(f"Cleaned rows: {len(clean_df)}")
    print(
        "Categories: "
        f"{clean_df['category'].nunique()}"
    )

    print()
    print("Cleaned column data types:")
    print(
        clean_df[
            [
                "price_gbp",
                "rating",
                "in_stock",
                "price_inr",
            ]
        ].dtypes
    )

    print()
    print(f"SQLite database: {DB_PATH}")
    print(f"SQL output: {SQL_OUTPUT_PATH}")
    print(
        "Pandas comparison output: "
        f"{PANDAS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()