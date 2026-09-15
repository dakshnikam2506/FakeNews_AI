import sys
import os


# Add src directory
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)


from predictor import predict_news


# ============================================================
# TEST CASES
# ============================================================

test_cases = [

    """
    The government announced a new national economic policy
    aimed at supporting businesses and increasing employment.
    """,

    """
    Scientists published a new study examining the effects
    of climate change on global temperatures.
    """,

    """
    BREAKING: Scientists confirm humans can survive without
    oxygen for 30 days. Doctors are hiding this discovery.
    """
]


# ============================================================
# RUN TESTS
# ============================================================

for i, text in enumerate(
    test_cases,
    start=1
):

    print("\n")
    print("=" * 60)
    print(f"TEST {i}")
    print("=" * 60)

    try:

        result = predict_news(text)

        print(
            "\nFINAL PREDICTION:",
            result["prediction"]
        )

        print(
            "CONFIDENCE:",
            result["confidence"]
        )

        print(
            "\nMODEL PREDICTIONS:"
        )

        for model, prediction in result[
            "model_predictions"
        ].items():

            print(
                f"  {model}: {prediction}"
            )

        print(
            "\nVOTES:",
            result["votes"]
        )

        print(
            "SVM SCORE:",
            result["svm_decision_score"]
        )

    except Exception as e:

        print(
            "\nERROR:",
            repr(e)
        )