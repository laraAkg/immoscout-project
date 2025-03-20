import scrapy
import re

class ImmoscoutSpider(scrapy.Spider):
    name = 'immoscout_spider'
    allowed_domains = ['immoscout24.ch']
    start_urls = ['https://www.immoscout24.ch/de/immobilien/mieten/ort-zuerich?=pn=1',]

    def parse(self, response):
        page_number = int(response.url.split('pn=')[-1])
        listings = response.css("div.ResultList_listItem_j5Td_")

        self.logger.info(f"Scraping Seite {page_number} - {len(listings)} Listings gefunden.")

        if not listings:
            self.logger.info(f"Seite {page_number} enthält keine Listings mehr. Beende den Crawl.")
            return

        # Datenextraktion aus Listings
        for listing in listings:
            rooms = listing.css('strong:nth-of-type(1)::text').get()
            size = listing.css('strong[title]::text').get()
            price = listing.css('span.HgListingRoomsLivingSpacePrice_price_u9Vee:nth-of-type(3)::text').get()
            location = listing.css('address::text').get()

            # Bereinigen der Daten
            rooms = self.clean_rooms(rooms)
            size = self.clean_size(size)
            price = self.clean_price(price)
            canton = response.url.split('-')[-1].split('?')[0]  # Kanton aus der URL extrahieren
            postal_code = self.extract_postal_code(location)  # Postleitzahl aus Adresse extrahieren

            yield {
                'rooms': rooms,
                'size': size,
                'price': price,
                'location': location.strip() if location else None,
                'canton': canton,
                'postal_code': postal_code,
                'page': response.url
            }

        next_page = page_number + 1
        canton = response.url.split('-')[-1].split('?')[0]
        next_page_url = f'https://www.immoscout24.ch/de/immobilien/mieten/ort-{canton}?pn={next_page}'
        
        yield scrapy.Request(next_page_url, callback=self.parse)

    def clean_rooms(self, rooms):
        if rooms:
            match = re.search(r'(\d+)', rooms)
            if match:
                return int(match.group(1))
        return None

    def clean_size(self, size):
        if size:
            match = re.search(r'(\d+)', size)
            if match:
                return int(match.group(1))
        return None

    def clean_price(self, price):
        if price:
            price = price.replace('CHF', '').replace('’', '').replace('–', '').strip()
            try:
                return float(price.replace(',', '.'))
            except ValueError:
                self.logger.warning(f"Preis konnte nicht konvertiert werden: {price}")
                return None
        return None

    def extract_postal_code(self, location):
        if location:
            match = re.search(r'(\d{4})', location)
            if match:
                return match.group(1)
        return None
