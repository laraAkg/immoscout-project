"""
This module contains a Scrapy spider for scraping property
listings from the ImmoScout24 website. The spider extracts
details such as the number of rooms, size, price, location,
canton, and postal code from each listing."""

import scrapy
import re


class ImmoscoutSpider(scrapy.Spider):
    name = "immoscout_spider"
    allowed_domains = ["immoscout24.ch"]
    start_urls = [
        "https://www.immoscout24.ch/de/immobilien/mieten/ort-zuerich?=pn=1",
    ]

    def parse(self, response):
        """
        Parses the response from the current page of the ImmoScout24 website and extracts property listings.
        This method processes the HTML response to extract details about property listings such as the number
        of rooms, size, price, location, canton, and postal code. It also handles pagination by generating
        requests for the next page if listings are found.
        Args:
            response (scrapy.http.Response): The HTTP response object containing the page's HTML content.
        Yields:
            dict: A dictionary containing the extracted property details:
                - 'rooms' (str or None): The number of rooms in the property, cleaned and formatted.
                - 'size' (str or None): The size of the property in square meters, cleaned and formatted.
                - 'price' (str or None): The price of the property, cleaned and formatted.
                - 'location' (str or None): The location of the property, stripped of extra whitespace.
                - 'canton' (str): The canton extracted from the URL.
                - 'postal_code' (str or None): The postal code extracted from the location.
                - 'page' (str): The URL of the current page being scraped.
        Yields:
            scrapy.Request: A request object for the next page of listings if more pages are available.
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

            yield {
                "rooms": rooms,
                "size": size,
                "price": price,
                "location": location.strip() if location else None,
                "canton": canton,
                "postal_code": postal_code,
                "page": response.url,
            }

        next_page = page_number + 1
        canton = response.url.split("-")[-1].split("?")[0]
        next_page_url = f"https://www.immoscout24.ch/de/immobilien/mieten/ort-{canton}?pn={next_page}"

        yield scrapy.Request(next_page_url, callback=self.parse)

    def clean_rooms(self, rooms):
        """
        Cleans and extracts the number of rooms from a given string.

        Args:
            rooms (str): A string containing room information, typically with numeric values.

        Returns:
            int: The extracted number of rooms as an integer if found, otherwise None.
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
        from the input price string, then attempts to convert it into a float.
        If the conversion fails, a warning is logged, and None is returned.

        Args:
            price (str): The price string to be cleaned and converted.

        Returns:
            float or None: The cleaned price as a float, or None if the input is
            invalid or cannot be converted.
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
            location (str): A string containing the location information.

        Returns:
            str or None: The extracted 4-digit postal code if found, otherwise None.
        """
        if location:
            match = re.search(r"(\d{4})", location)
            if match:
                return match.group(1)
        return None
