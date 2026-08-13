# Module 1 — Data Pipeline

This module implements the scraping, cleaning, currency conversion,
SQLite loading, SQL querying, and pandas querying requirements for
Module 1 of the capstone project.

## Data Source

The data is scraped from:

https://books.toscrape.com/

The scraper processes complete book categories and continues until the
cleaned dataset contains at least 60 books across at least 3 categories.

For every book, the following source fields are collected:

- title
- price
- star_rating
- availability
- category

## Installation

Install the required Python packages:

```bash
pip install requests beautifulsoup4 pandas