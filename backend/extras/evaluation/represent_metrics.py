import matplotlib.pyplot as plt

def plot_metrics():
    # Metrics
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [86.46, 82.11, 84.23]

    # Neutral colors (soft greys/blues)
    colors = ['#4C566A', '#5E81AC', '#88C0D0']

    # Create bar chart
    plt.figure()
    bars = plt.bar(metrics, values, color=colors)

    # Add value labels on top of each bar
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{value:.2f}%',
                 ha='center')

    # Visual settings
    plt.ylim(0, 100)
    plt.ylabel('Percentage (%)')
    plt.title('Model Performance Metrics')

    # Clean look (remove top/right borders)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.show()

# plot_metrics()


import pandas as pd
import webbrowser
import os
import tempfile
import html

def show_cv_table_web(
    file_path=r"C:\Users\danie\Desktop\My_Things\Hobbies\VSCODE\WebSites\cv_analyser_website\backend\outputs\all_results.csv",
    selected_columns=None,
    max_text_length=90
):
    """
    Load CV analysis results from a CSV file, create a polished HTML table,
    and open it in the default browser.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    selected_columns : list[str] | None
        Columns to display. If None, a default presentation-friendly set is used.
    max_text_length : int
        Maximum length for long text fields before truncation.
    """

    default_columns = [
        "person_name",
        "job_title",
        "company",
        "university",
        "degree",
        "skills",
        "experience_years",
        "email",
        "phone",
        "languages",
        "linkedin",
        "github",
    ]

    column_labels = {
        "person_name": "Candidate",
        "job_title": "Current / Recent Role",
        "company": "Company / Organisation",
        "university": "Education Institution",
        "degree": "Academic Background",
        "skills": "Key Skills                            .",
        "experience_years": "Experience",
        "email": "Email",
        "phone": "Phone Number      .",
        "languages": "Languages",
        "linkedin": "LinkedIn",
        "github": "GitHub",
    }

    if selected_columns is None:
        selected_columns = default_columns

    # Read everything as string to avoid phone numbers becoming floats
    df = pd.read_csv(file_path, sep=";", dtype=str).fillna("")

    # Keep only columns that exist in the file
    available_columns = [col for col in selected_columns if col in df.columns]
    df = df[available_columns].copy()

    def shorten_text(value, max_len=max_text_length):
        value = str(value).strip()
        if len(value) > max_len:
            return value[:max_len - 3] + "..."
        return value

    def format_phone(phone):
        phone = str(phone).strip().replace(".0", "").replace(" ", "")
        if not phone:
            return ""

        # Keep explicit international format if present
        if phone.startswith("+"):
            digits = "+" + "".join(ch for ch in phone[1:] if ch.isdigit())
        else:
            digits = "".join(ch for ch in phone if ch.isdigit())

        # Portugal pattern: +351XXXXXXXXX or 351XXXXXXXXX
        if digits.startswith("+351") and len(digits) >= 13:
            raw = digits[1:]
            return f"+{raw[:3]} {raw[3:6]} {raw[6:9]} {raw[9:12]}"
        if digits.startswith("351") and len(digits) >= 12:
            return f"+{digits[:3]} {digits[3:6]} {digits[6:9]} {digits[9:12]}"

        return phone.replace(".0", "")

    def make_email_link(email_value):
        email_value = str(email_value).strip()
        if not email_value:
            return ""
        safe_email = html.escape(email_value)
        return f'<a href="mailto:{safe_email}">{safe_email}</a>'

    def make_url_link(url_value, label):
        url_value = str(url_value).strip()
        if not url_value:
            return ""
        safe_url = html.escape(url_value)
        return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{label}</a>'

    def clean_text(value):
        return html.escape(str(value).strip())

    # Apply formatting column by column
    if "phone" in df.columns:
        df["phone"] = df["phone"].apply(format_phone)

    if "email" in df.columns:
        df["email"] = df["email"].apply(make_email_link)

    if "linkedin" in df.columns:
        df["linkedin"] = df["linkedin"].apply(lambda x: make_url_link(x, "View Profile"))

    if "github" in df.columns:
        df["github"] = df["github"].apply(lambda x: make_url_link(x, "View GitHub"))

    # Escape and shorten text columns, but keep HTML columns unescaped
    html_columns = {"email", "linkedin", "github"}

    for col in df.columns:
        if col in html_columns:
            continue
        df[col] = df[col].apply(clean_text).apply(shorten_text)

    # Rename columns for presentation
    df = df.rename(columns={col: column_labels.get(col, col) for col in df.columns})

    table_html = df.to_html(index=False, classes="cv-table", escape=False)

    html_page = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CV Analysis Overview</title>
        <style>
            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
                background: linear-gradient(180deg, #f4f6f8 0%, #eef2f6 100%);
                color: #243447;
                padding: 32px;
            }}

            .page {{
                max-width: 1600px;
                margin: 0 auto;
            }}

            .header {{
                margin-bottom: 20px;
            }}

            .title {{
                margin: 0;
                font-size: 30px;
                font-weight: 700;
                color: #2f3b52;
                letter-spacing: -0.02em;
            }}

            .subtitle {{
                margin-top: 8px;
                color: #667085;
                font-size: 15px;
            }}

            .card {{
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #e6eaf0;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(31, 41, 55, 0.08);
                overflow: hidden;
                backdrop-filter: blur(8px);
            }}

            .table-wrapper {{
                overflow-x: auto;
                overflow-y: auto;
                max-height: 78vh;
            }}

            table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                min-width: 1300px;
            }}

            thead th {{
                position: sticky;
                top: 0;
                z-index: 2;
                background: #4c566a;
                color: #ffffff;
                text-align: left;
                font-size: 14px;
                font-weight: 600;
                padding: 16px 14px;
                border-bottom: 1px solid #3f4758;
                white-space: nowrap;
            }}

            tbody td {{
                padding: 16px 14px;
                font-size: 14px;
                line-height: 1.45;
                color: #2d3748;
                border-bottom: 1px solid #e7ecf2;
                vertical-align: top;
                background: #ffffff;
            }}

            tbody tr:nth-child(even) td {{
                background: #f8fafc;
            }}

            tbody tr:hover td {{
                background: #eef3f8;
                transition: background 0.2s ease;
            }}

            th:first-child {{
                border-top-left-radius: 16px;
            }}

            th:last-child {{
                border-top-right-radius: 16px;
            }}

            a {{
                color: #486581;
                text-decoration: none;
                font-weight: 500;
            }}

            a:hover {{
                text-decoration: underline;
            }}

            .footer-note {{
                margin-top: 14px;
                font-size: 12px;
                color: #7b8794;
            }}

            @media (max-width: 900px) {{
                body {{
                    padding: 16px;
                }}

                .title {{
                    font-size: 24px;
                }}

                thead th,
                tbody td {{
                    padding: 12px 10px;
                    font-size: 13px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="page">
            <div class="header">
                <h1 class="title">CV Analysis Overview</h1>
                <div class="subtitle">
                    Presentation-friendly candidate summary with selected profile fields.
                </div>
            </div>

            <div class="card">
                <div class="table-wrapper">
                    {table_html}
                </div>
            </div>

            <div class="footer-note">
                Generated automatically from CSV data.
            </div>
        </div>
    </body>
    </html>
    """

    output_file = os.path.join(tempfile.gettempdir(), "cv_analysis_overview.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_page)

    webbrowser.open(f"file://{output_file}")

show_cv_table_web()