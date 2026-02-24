# We create const of node names to avoid code duplication
# Everytime we reference the nodes, we reference the consts
# If we change the node name, we only need to change it one one place (here)
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEB_SEARCH="web_search"