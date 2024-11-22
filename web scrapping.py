!pip install scrapy import scrapy
from scrapy.crawler import CrawlerProcess

class QuotesSpider(scrapy.Spider):name
= 'quotes'
start_urls = ['http://quotes.toscrape.com/']

def parse(self, response):

for quote in response.css('div.quote'): text = quote.css('span.text::text').get()author = quote.css('small::text').get()
print(f'Text: {text}\nAuthor: {author}\n{"-"*40}')

next_page = response.css('li.next a::attr(href)').get()if next_page:
yield response.follow(next_page, self.parse)

if		name	== "	main	": process = CrawlerProcess({

'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',

})
 

process.crawl(QuotesSpider) process.start()
