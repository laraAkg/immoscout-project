import scrapy

class ImmoscoutSpider(scrapy.Spider):
    name = 'immoscout_spider'
    allowed_domains = ['immoscout24.ch']
    start_urls = ['https://www.immoscout24.ch/de/immobilien/mieten/ort-zuerich?pn=1']

    def parse(self, response):
        page_number = int(response.url.split('pn=')[-1])
        listings = response.css('[data-test="listing-card-large"]')

        self.logger.info(f"Scraping Seite {page_number} - {len(listings)} Listings gefunden.")

        if not listings:
            self.logger.info(f"Seite {page_number} enthält keine Listings mehr. Beende den Crawl.")
            return

        for listing in listings:
            rooms = listing.css('strong:nth-of-type(1)::text').get()
            size = listing.css('strong[title]::text').get()
            price = listing.css('span.HgListingRoomsLivingSpacePrice_price_u9Vee:nth-of-type(3)::text').get()
            location = listing.css('address::text').get()

            yield {
                'rooms': rooms.strip() if rooms else None,
                'size': size.strip() if size else None,
                'price': price.strip() if price else None,
                'location': location.strip() if location else None,
                'page': response.url
            }

        # Nächste Seite, nur wenn Listings da waren
        next_page = page_number + 1
        next_page_url = f'https://www.immoscout24.ch/de/immobilien/mieten/ort-zuerich?pn={next_page}'
        yield scrapy.Request(next_page_url, callback=self.parse)
