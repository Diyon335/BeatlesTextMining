from fastcoref import FCoref


def run_coref():
    model = FCoref()

    preds = model.predict(
        texts=['We are so happy to see you using our coref package. This package is very fast!']
    )

    print(preds[0].get_clusters(as_strings=False))

    print(preds[0].get_clusters())

    print(preds[0].get_logit(
        span_i=(33, 50), span_j=(52, 64)
    ))



