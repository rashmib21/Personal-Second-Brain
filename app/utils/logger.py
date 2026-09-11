import logging
import sys #use to print logs in terminals

# Suppress HTTP request logs from external HTTP clients (httpx, httpcore, urllib3)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

#create a logger object
logger=logging.getLogger()

#show info, warning, error, and critical msg
logger.setLevel(logging.INFO)

#print all logs in the terminals
handler=logging.StreamHandler(sys.stdout)

#set the format of logs in which pattern they will be print
formatter=logging.Formatter(
	"%(asctime)s | %(levelname)s | %(message)s")
#asctime: time, levelname: info, error, warning, message: log message like Scanner started

#attach the formatter to the handler, every log printed by this handler 
handler.setFormatter(formatter)

#attach the handler to the logger, logger knows where to send logs and how to format them
logger.addHandler(handler)

#use the logger
logger.info("Scanner started...")
logger.warning("Folder is empty...")
logger.error("Redis connection failed.. try again...")



