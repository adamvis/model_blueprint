import pandas as pd
from transformers import CustomTransformer
from reporting import Reporter, convert_md_to_pdf
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline


if __name__ == "__main__":

    # Example usage
    base_dir = "models"
    model_name = "example_model"
    reporter = Reporter(base_dir, model_name)

    # Create an instance of CustomTransformer with logging
    transformer_a = CustomTransformer(reporter=reporter)
    transformer_b = CustomTransformer(reporter=reporter)

    # Sample data
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    execution_pipeline = Pipeline(
        steps = [
            ("first_exec", transformer_a),
            ("second_exec", transformer_b),
        ]
    )

    # Fit the transformer (logging happens during training)
    execution_pipeline.fit(df)

    # Save logs to markdown
    md_file_path = reporter.save_to_markdown()

    # Convert markdown to PDF
    convert_md_to_pdf(md_file_path)