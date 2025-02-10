import logging


def decide_to_generate(state):

    logging.info("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        logging.info("---DECISION: WEB SEARCH---")
        return "transform_query"
    else:
        logging.info("---DECISION: GENERATE---")
        return "generate"
