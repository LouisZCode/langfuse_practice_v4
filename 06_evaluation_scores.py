import os
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()
langfuse = get_client()

'''
langfuse.create_score(
    trace_id="2b7167ecfcaa736b25a28e546cbc56c2",
    name="my-first-score",
    value=0.9,
    data_type="NUMERIC",
    comment="Looks good to me",
)
'''

langfuse.create_score(
    trace_id="2b7167ecfcaa736b25a28e546cbc56c2",
    observation_id="8161783405a0e55a",  # the generation's ID, not the trace's
    name="response-quality",
    value=0.7,
    data_type="NUMERIC",
)
langfuse.flush()