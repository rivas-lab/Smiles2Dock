import logging

logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=logging.INFO)

def print_block():
    logging.info('##################################################################')
