# Setting the API Key as an Environment Variable

import os
from dotenv import load_dotenv

# Read key-value pairs from a local .env file and add them to the process
# environment. Keeping the key outside the source code prevents it from being
# committed to Git accidentally.
load_dotenv()

# os.getenv() returns None when the variable is missing. API examples later in
# this section pass this value to a client or let the client read the same
# environment variable automatically.
api_key = os.getenv("OPENAI_API_KEY")

# Summary: store OPENAI_API_KEY in .env, call load_dotenv() once at startup,
# and retrieve the value with os.getenv(). Never print or commit the key.
