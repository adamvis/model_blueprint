import os
import logging
import pandas as pd
from io import StringIO
from datetime import datetime
import pdfkit

md_title_levels = {
    1: "#",
    2: "##",
    3: "###",
    4: "####",
    5: "#####",
}


class Reporter:
    def __init__(self, base_dir, model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
        os.makedirs(self.dir, exist_ok=True)
        self.buffer = StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.logger = logging.getLogger('CustomLogger')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def log(self, message):
        self.logger.info(message + "\n")

    def log_table(self, df, caption=""):
        table_md = df.to_markdown()
        full_md = f"{caption}\n\n{table_md}\n"
        self.logger.info(full_md)

    def log_plot(self, fig, caption=""):
        fig_file = os.path.join(self.dir, f"fig_{len(os.listdir(self.dir))}.png")
        fig.savefig(fig_file)
        fig_md = f"![{caption}]({fig_file})\n"
        self.logger.info(fig_md)

    def get_logs(self):
        self.handler.flush()
        return self.buffer.getvalue()

    def save_to_markdown(self, md_file='execution_report.md'):
        buffer_content = self.get_logs()
        md_file_path = os.path.join(self.dir, md_file)
        with open(md_file_path, 'w') as mf:
            mf.write(buffer_content)
        return md_file_path

def convert_md_to_pdf(md_file_path, pdf_file='execution_report.pdf'):
    pdf_file_path = md_file_path.replace('.md', '.pdf')
    pdfkit.from_file(md_file_path, pdf_file_path)
    return pdf_file_path


