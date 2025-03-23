"""
This module defines a Scrapy spider for scraping real estate listings from the 
ImmoScout24 website. The spider extracts property details such as the number of 
rooms, size, price, location, canton, and postal code, and stores the data in a 
MongoDB database. It also handles pagination to scrape multiple pages of listings.

Classes:
    ImmoscoutSpider(scrapy.Spider): A Scrapy spider for crawling and extracting 
    real estate listings from ImmoScout24.

Methods:
    __init__(*args, **kwargs):
        Initializes the spider, sets up a MongoDB connection, and clears the 
        existing database collection.

    parse(response):
        Parses the response from the current page, extracts property listings, 
        stores them in the database, and handles pagination.

    clean_rooms(rooms):

    clean_size(size):

    clean_price(price):

    extract_postal_code(location):

    closed(reason):
        Ensures the MongoDB client connection is closed when the spider finishes 
        execution.

"""

import scrapy
import re
import os
from pymongo import MongoClient
from scrapy.utils.project import get_project_settings

class ImmoscoutSpider(scrapy.Spider):
    name = "immoscout_spider"
    allowed_domains = ["immoscout24.ch"]
    start_urls = [
        "https://www.immoscout24.ch/de/immobilien/mieten/kanton-zuerich?pn=1",
    ]

    def __init__(self, *args, **kwargs):
        """
        Initializes the spider with necessary configurations and sets up a connection
        to the MongoDB database. Ensures that the `MONGO_URI` environment variable is
        set, and raises an error if it is not. Connects to the MongoDB instance, selects
        the `immoscout_db` database, and clears all entries in the `listings` collection.

        Raises:
            ValueError: If the `MONGO_URI` environment variable is not set.
        """
        super().__init__(*args, **kwargs)
        settings = get_project_settings()
        MONGO_URI = os.getenv("MONGO_URI")

        if not MONGO_URI:
            raise ValueError(
                "❌ MONGO_URI ist nicht gesetzt! Bitte ENV-Variable oder Secret einrichten."
            )

        self.client = MongoClient(MONGO_URI)
        self.db = self.client["immoscout_db"]
        self.collection = self.db["listings"]
        self.collection.delete_many({})
        self.logger.info("Alle alten Einträge in der MongoDB wurden gelöscht.")

    def parse(self, response):
        """
        Parses the response from the current page, extracts property listings, and stores them in the database.
        Also handles pagination by generating a request for the next page.

        Args:
            response (scrapy.http.Response): The response object containing the HTML content of the current page.

        Extracted Data:
            - rooms (int): Number of rooms in the property.
            - size (float): Size of the property in square meters.
            - price (float): Price of the property.
            - location (str): Address or location of the property.
            - canton (str): Canton extracted from the URL.
            - postal_code (str): Postal code extracted from the location.
            - page (str): URL of the current page.

        Behavior:
            - Logs the current page number and the number of listings found.
            - If no listings are found, logs a message and stops the crawl.
            - Cleans and processes extracted data before storing it in the database.
            - Constructs the URL for the next page and yields a new request to continue crawling.

        Yields:
            scrapy.Request: A request object for the next page to be crawled.
        """
        page_number = int(response.url.split("pn=")[-1])
        listings = response.css("div.ResultList_listItem_j5Td_")
        self.logger.info(
            f"Scraping Seite {page_number} - {len(listings)} Listings gefunden."
        )

        if not listings:
            self.logger.info(
                f"Seite {page_number} enthält keine Listings mehr. Beende den Crawl."
            )
            return

        for listing in listings:
            rooms = listing.css("strong:nth-of-type(1)::text").get()
            size = listing.css("strong[title]::text").get()
            price = listing.css(
                "span.HgListingRoomsLivingSpacePrice_price_u9Vee:nth-of-type(3)::text"
            ).get()
            location = listing.css("address::text").get()
            rooms = self.clean_rooms(rooms)
            size = self.clean_size(size)
            price = self.clean_price(price)
            canton = response.url.split("-")[-1].split("?")[0]
            postal_code = self.extract_postal_code(location)
            self.collection.insert_one(
                {
                    "rooms": rooms,
                    "size": size,
                    "price": price,
                    "location": location.strip() if location else None,
                    "canton": canton,
                    "postal_code": postal_code,
                    "page": response.url,
                }
            )

        next_page = page_number + 1
        canton = response.url.split("-")[-1].split("?")[0]
        next_page_url = f"https://www.immoscout24.ch/de/immobilien/mieten/kanton-zuerich?pn={next_page}"

        yield scrapy.Request(next_page_url, callback=self.parse)

    def clean_rooms(self, rooms):
        """
        Cleans and extracts the number of rooms from a given string.

        This method searches for the first occurrence of one or more digits
        in the input string and returns it as an integer. If no digits are
        found or the input is None, it returns None.

        Args:
            rooms (str): A string potentially containing the number of rooms.

        Returns:
            int or None: The extracted number of rooms as an integer, or None
            if no valid number is found.
        """
        if rooms:
            match = re.search(r"(\d+)", rooms)
            if match:
                return int(match.group(1))
        return None

    def clean_size(self, size):
        """
        Extracts and cleans the numeric value representing size from a given string.

        Args:
            size (str): A string potentially containing a numeric size value.

        Returns:
            int or None: The extracted numeric size as an integer if found, 
            otherwise None.
        """
        if size:
            match = re.search(r"(\d+)", size)
            if match:
                return int(match.group(1))
        return None

    def clean_price(self, price):
        """
        Cleans and converts a price string into a float.

        This method removes currency symbols, special characters, and whitespace 
        from the input price string. It then attempts to convert the cleaned string 
        into a float. If the conversion fails, a warning is logged, and None is returned.

        Args:
            price (str): The price string to be cleaned and converted.

        Returns:
            float or None: The cleaned and converted price as a float, or None if 
            the input is invalid or conversion fails.
        """
        if price:
            price = price.replace("CHF", "").replace("’", "").replace("–", "").strip()
            try:
                return float(price.replace(",", "."))
            except ValueError:
                self.logger.warning(f"Preis konnte nicht konvertiert werden: {price}")
                return None
        return None

    def extract_postal_code(self, location):
        """
        Extracts the postal code from a given location string.

        Args:
            location (str): The location string that may contain a postal code.

        Returns:
            str or None: The extracted 4-digit postal code if found, otherwise None.
        """
        if location:
            match = re.search(r"(\d{4})", location)
            if match:
                return match.group(1)
        return None

    def closed(self, reason):
        """
        Called when the spider is closed.

        This method is triggered automatically when the spider finishes its execution,
        either successfully or due to an error. It ensures that the database client
        connection is properly closed to release resources.

        Args:
            reason (str): The reason why the spider was closed. This could be 'finished',
                          'cancelled', or an error message.
        """
        self.client.close()