import sys
from forecasting_tools import BinaryQuestion, GeneralLlm
from bots import PerplexityRelatedMarketsBot
import asyncio
import os
import webbrowser
from datetime import datetime
import markdown


def main():
    print("Welcome to the Metaculus Bot CLI!")
    question = input("Please enter a question for the bot to forecast on: ")
    # Create a pretend BinaryQuestion object
    user_question = BinaryQuestion(
        question_text=question,
        resolution_criteria="",
        fine_print=None,
        background_info=None,
        id_of_post=-1,
        page_url="cli://user-question"
    )

    bot = PerplexityRelatedMarketsBot(
        llms={
            "default": GeneralLlm(model="openrouter/openai/gpt-4o-mini", temperature=0.2),
            "summarizer": GeneralLlm(model="openrouter/openai/gpt-4o-mini", temperature=0.2)
        }
    )

    print("Running forecast... (this may take a moment)")
    forecast_reports = asyncio.run(
        bot.forecast_questions([user_question], return_exceptions=True)
    )
    print("\nForecast result:")
    for report in forecast_reports:
        print(report)
        # Try to extract the markdown explanation
        explanation = getattr(report, 'explanation', None)
        if explanation:
            # Convert markdown to HTML
            html = markdown.markdown(explanation, extensions=[
                                     'extra', 'tables', 'sane_lists'])
            # Add a simple HTML wrapper
            html_content = f"""
            <html>
            <head>
                <meta charset='utf-8'>
                <title>Forecast Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    pre, code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
                    h1, h2, h3, h4 {{ color: #2c3e50; }}
                </style>
            </head>
            <body>
            {html}
            </body>
            </html>
            """
            # Prepare output directory and filename
            output_dir = "local_forecasts"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"forecast-{timestamp}.html"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"\nForecast report written to: {filepath}")
            print("Opening in browser...")
            webbrowser.open(f"file://{os.path.abspath(filepath)}")


if __name__ == "__main__":
    main()
