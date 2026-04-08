from graph_chat.log_utils import log


# ... existing code ...
def draw_graph(graph, file_name: str):
    try:
        mermaid_code = graph.get_graph(xray=True).draw_mermaid_png()
        with open(file_name, "wb") as f:
            f.write(mermaid_code)

    except Exception as e:
        log.exception(e)
