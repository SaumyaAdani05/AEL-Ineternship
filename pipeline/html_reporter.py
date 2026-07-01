"""
html_reporter.py — Dashboard HTML Report Generator.

Writes premium, self-contained HTML dashboards to display the model results
with Tailwind CSS styling and Chart.js interactive charts.
"""
import json
import os
import pandas as pd
from typing import Dict, List, Any

from pipeline import cli_formatter as cf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "reports", "generated")

# Common Premium Header with navigation or branding
HTML_HEAD = """
<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-950 text-slate-100">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts: Inter & DM Sans -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
        }}
        .font-title {{
            font-family: 'DM Sans', sans-serif;
        }}
        .font-mono-data {{
            font-family: 'JetBrains Mono', monospace;
        }}
    </style>
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    colors: {{
                        brand: {{
                            50: '#f0f9ff',
                            100: '#e0f2fe',
                            500: '#0ea5e9',
                            600: '#0284c7',
                            700: '#0369a1',
                            950: '#070a13',
                        }}
                    }}
                }}
            }}
        }}
    </script>
</head>
<body class="h-full flex flex-col font-sans">
"""

def generate_xgboost_html_report(
    risk_table: pd.DataFrame,
    importance_dict: Dict[str, float],
    train_cindex: float,
    test_cindex: float,
    output_path: str = None
) -> None:
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, "report_xgboost.html")
    """
    Generate an interactive executive HTML dashboard for the XGBoost Survival model.
    """
    # 1. Format employee risk data
    col_headers = list(risk_table.columns)
    risk_data = []
    high_risk_count = 0
    
    # Sort by 12-month risk descending
    twelve_month_col = col_headers[-1]
    sorted_table = risk_table.sort_values(twelve_month_col, ascending=False)
    
    for idx, row in sorted_table.iterrows():
        # EMP ID from index or sequential index mapping
        emp_num = len(risk_data) + 1
        emp_id = f"EMP_{emp_num:04d}"
        
        is_high_risk = any(row[col] >= 0.30 for col in col_headers)
        if is_high_risk:
            high_risk_count += 1
            
        risk_data.append({
            "id": emp_id,
            "risk_1m": f"{row[col_headers[0]] * 100:.1f}%",
            "risk_3m": f"{row[col_headers[1]] * 100:.1f}%",
            "risk_6m": f"{row[col_headers[2]] * 100:.1f}%",
            "risk_12m": f"{row[col_headers[3]] * 100:.1f}%",
            "high_risk": is_high_risk
        })

    # 2. Format feature importance data
    imp_series = pd.Series(importance_dict).sort_values(ascending=False).head(10)
    features = []
    gains = []
    for feat, gain in imp_series.items():
        # Human readable names
        n = feat.replace("_", " ").title()
        if "Overtime" in n:
            n = "Working Overtime"
        elif "Maritalstatus" in n:
            n = n.replace("Maritalstatus", "Marital Status:")
        elif "Department" in n:
            n = n.replace("Department", "Department:")
        elif "Gender" in n:
            n = n.replace("Gender", "Gender:")
        features.append(n)
        gains.append(float(gain))

    # Identify top driver
    top_driver = features[0] if features else "N/A"
    
    # Determine recommendation
    recommendation_title = "RETENTION LEADERSHIP ACTION"
    recommendation_class = "border-sky-500/30 bg-sky-950/20 text-sky-200"
    recommendation_badge = "bg-sky-500/20 text-sky-400 border-sky-500/30"
    recommendation_text = (
        f"The primary driver of attrition is <strong>{top_driver}</strong>. "
        "Compensation-related features are not in the top 3 drivers. "
        "Focus on improving leadership quality, manager relations, and team culture before adjusting base compensation."
    )
    
    # Check if compensation is a top driver
    income_words = ["income", "salary", "rate", "daily", "monthly", "hourly"]
    top_3_feats = list(imp_series.head(3).index)
    if any(any(w in f.lower() for w in income_words) for f in top_3_feats):
        recommendation_title = "CRITICAL COMPENSATION ACTION"
        recommendation_class = "border-rose-500/30 bg-rose-950/20 text-rose-200"
        recommendation_badge = "bg-rose-500/20 text-rose-400 border-rose-500/30"
        recommendation_text = (
            "Pay MATTERS here. Compensation-related features are within the top 3 risk drivers. "
            "Review salary bands, overtime policies, and role adjustments immediately for high-turnover departments."
        )

    # Compile HTML
    html_content = HTML_HEAD.format(title="XGBoost Attrition Risk Intelligence Dashboard")
    
    html_content += f"""
    <!-- Main Content Grid -->
    <main class="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 max-w-7xl mx-auto w-full">
        
        <!-- Dashboard Header Title -->
        <div class="border-l-4 border-sky-500 pl-4 py-2">
            <h1 class="text-3xl font-title font-bold tracking-tight text-white">Employee Attrition Risk Assessment</h1>
            <p class="text-slate-400 mt-1">Deep predictive analytics powered by machine learning to pinpoint attrition drivers and flight risks.</p>
        </div>

        <!-- Row 1: KPI Stats Grid -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <!-- Card 1 -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm hover:border-slate-700 transition duration-200">
                <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Total Test Population</p>
                <h3 class="text-3xl font-mono-data font-bold mt-2 text-white">{len(risk_table)}</h3>
                <p class="text-xs text-slate-500 mt-1">Employees assessed in test cohort</p>
            </div>
            <!-- Card 2 -->
            <div class="bg-slate-900 border border-slate-850 rounded-xl p-5 shadow-sm hover:border-slate-700 transition duration-200">
                <p class="text-xs text-rose-400 font-medium uppercase tracking-wider">High Risk Cohort</p>
                <div class="flex items-baseline space-x-2">
                    <h3 class="text-3xl font-mono-data font-bold mt-2 text-rose-500">{high_risk_count}</h3>
                    <span class="text-sm font-semibold text-rose-400">({high_risk_count/len(risk_table)*100:.1f}%)</span>
                </div>
                <p class="text-xs text-slate-500 mt-1">Attrition probability &ge; 30%</p>
            </div>
            <!-- Card 3 -->
            <div class="bg-slate-900 border border-slate-850 rounded-xl p-5 shadow-sm hover:border-slate-700 transition duration-200">
                <p class="text-xs text-emerald-400 font-medium uppercase tracking-wider">Model Accuracy (C-Index)</p>
                <h3 class="text-3xl font-mono-data font-bold mt-2 text-emerald-500">{test_cindex * 100:.1f}%</h3>
                <p class="text-xs text-slate-500 mt-1">Generalization on unseen data</p>
            </div>
            <!-- Card 4 -->
            <div class="bg-slate-900 border border-slate-850 rounded-xl p-5 shadow-sm hover:border-slate-700 transition duration-200">
                <p class="text-xs text-sky-400 font-medium uppercase tracking-wider">Top Risk Driver</p>
                <h3 class="text-lg font-bold mt-3 text-white truncate">{top_driver}</h3>
                <p class="text-xs text-slate-500 mt-2">Ranked by information gain</p>
            </div>
        </div>

        <!-- Row 2: Executive decision recommendations -->
        <div class="border rounded-xl p-6 {recommendation_class}">
            <div class="flex items-center space-x-3 mb-3">
                <span class="text-xs uppercase font-bold tracking-wider px-3 py-1 border rounded-full {recommendation_badge}">
                    {recommendation_title}
                </span>
            </div>
            <p class="text-sm font-medium leading-relaxed">{recommendation_text}</p>
        </div>

        <!-- Row 3: Charts Column & False Alarms Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Left Chart Card (2 Columns wide) -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm lg:col-span-2 space-y-4">
                <h3 class="text-lg font-title font-bold text-white">XGBoost Attrition Drivers (Relative Gain)</h3>
                <div class="relative h-80">
                    <canvas id="featureChart"></canvas>
                </div>
            </div>

            <!-- Right Info Card (1 Column) -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm flex flex-col justify-between">
                <div>
                    <h3 class="text-lg font-title font-bold text-white mb-2">The False Alarms</h3>
                    <p class="text-xs text-slate-400 leading-relaxed mb-4">
                        These features have negligible importance. Do not waste company budget or execute policy changes addressing them:
                    </p>
                    <ul class="space-y-3">
    """
    
    # Render trailing features (False alarms)
    low_features = pd.Series(importance_dict).sort_values().head(5)
    for feat, gain in low_features.items():
        n = feat.replace("_", " ").title()
        html_content += f"""
                        <li class="flex items-center justify-between bg-slate-950/40 px-3 py-2 rounded border border-slate-800">
                            <span class="text-sm text-slate-300 font-medium">{n}</span>
                            <span class="text-xs font-mono-data text-yellow-500/80 bg-yellow-500/10 px-2 py-0.5 rounded border border-yellow-500/20">Gain: {gain:.1f}</span>
                        </li>
        """

    html_content += f"""
                    </ul>
                </div>
                <div class="border-t border-slate-800 mt-6 pt-4 text-xs text-slate-500 leading-normal">
                    💡 <em>Note: Low-impact features represent statistical noise, not root causes of turnover.</em>
                </div>
            </div>
        </div>

        <!-- Row 4: Risk profiles Interactive table -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
            <div class="p-6 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h3 class="text-lg font-title font-bold text-white">Employee Attrition Risk Profiles</h3>
                    <p class="text-xs text-slate-400 mt-1">Flight risk probabilities calculated across 1-Month, 3-Month, 6-Month, and 12-Month horizons.</p>
                </div>
                <!-- Filter Toolbar -->
                <div class="flex items-center gap-3">
                    <input type="text" id="searchEmp" onkeyup="onFilterChange()" placeholder="Search Employee ID..." class="px-3.5 py-1.5 bg-slate-950 border border-slate-800 focus:border-sky-500 focus:outline-none rounded-lg text-sm text-slate-100 placeholder-slate-500">
                    <select id="statusFilter" onchange="onFilterChange()" class="px-3.5 py-1.5 bg-slate-950 border border-slate-800 focus:border-sky-500 focus:outline-none rounded-lg text-sm text-slate-300 cursor-pointer">
                        <option value="all">All Employees</option>
                        <option value="high">High Risk Only</option>
                        <option value="normal">Normal Only</option>
                    </select>
                </div>

            </div>

            <!-- Table content container -->
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="border-b border-slate-800 bg-slate-950/20 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                            <th class="py-3 px-6">Employee ID</th>
                            <th class="py-3 px-6 text-right">1-Month</th>
                            <th class="py-3 px-6 text-right">3-Month</th>
                            <th class="py-3 px-6 text-right">6-Month</th>
                            <th class="py-3 px-6 text-right">12-Month</th>
                            <th class="py-3 px-6 text-center">Status</th>
                        </tr>
                    </thead>
                    <tbody id="employeeTableBody" class="divide-y divide-slate-800/60 text-sm font-medium">
    """

    for emp in risk_data:
        status_pill = (
            '<span class="px-2.5 py-1 text-xs rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 font-bold uppercase">● High Risk</span>'
            if emp["high_risk"] else
            '<span class="px-2.5 py-1 text-xs rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold uppercase">○ Normal</span>'
        )
        emp_class = "text-rose-400 font-bold" if emp["high_risk"] else "text-emerald-400"
        
        # Color specific cell highlights if high risk
        r1 = f'<span class="text-rose-400 font-bold">{emp["risk_1m"]}</span>' if emp["high_risk"] else emp["risk_1m"]
        r3 = f'<span class="text-rose-400 font-bold">{emp["risk_3m"]}</span>' if emp["high_risk"] else emp["risk_3m"]
        r6 = f'<span class="text-rose-400 font-bold">{emp["risk_6m"]}</span>' if emp["high_risk"] else emp["risk_6m"]
        r12 = f'<span class="text-rose-400 font-bold">{emp["risk_12m"]}</span>' if emp["high_risk"] else emp["risk_12m"]

        html_content += f"""
                        <tr data-high-risk="{"true" if emp["high_risk"] else "false"}" class="hover:bg-slate-800/30 transition duration-150">
                            <td class="py-3.5 px-6 font-mono-data {emp_class} emp-id">{emp["id"]}</td>
                            <td class="py-3.5 px-6 text-right font-mono-data">{r1}</td>
                            <td class="py-3.5 px-6 text-right font-mono-data">{r3}</td>
                            <td class="py-3.5 px-6 text-right font-mono-data">{r6}</td>
                            <td class="py-3.5 px-6 text-right font-mono-data">{r12}</td>
                            <td class="py-3.5 px-6 text-center">{status_pill}</td>
                        </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>
            
            <!-- Pagination Controls -->
            <div class="px-6 py-4 border-t border-slate-800 flex items-center justify-between bg-slate-950/20 text-sm text-slate-400">
                <span id="paginationInfo">Showing 1 to 10 of 298 entries</span>
                <div class="flex items-center space-x-2">
                    <button id="prevPageBtn" onclick="prevPage()" class="px-3 py-1.5 bg-slate-950 hover:bg-slate-850 hover:text-white border border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300 rounded-lg transition font-medium">Previous</button>
                    <button id="nextPageBtn" onclick="nextPage()" class="px-3 py-1.5 bg-slate-950 hover:bg-slate-850 hover:text-white border border-slate-800 disabled:opacity-40 disabled:cursor-not-allowed text-slate-300 rounded-lg transition font-medium">Next</button>
                </div>
            </div>
        </div>
    </main>


    <!-- Footer -->
    <footer class="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-600">
        &copy; 2026 MNC Enterprise Retention Systems. All Rights Reserved. Confidential - Internal Use Only.
    </footer>

    <!-- Interactive script bindings -->
    <script>
        // Draw Feature chart
        const ctx = document.getElementById('featureChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(features)},
                datasets: [{{
                    label: 'Gain Importance',
                    data: {json.dumps(gains)},
                    backgroundColor: [
                        '#ef4444', '#ef4444', '#ef4444', // Red for top 3
                        '#eab308', '#eab308', '#eab308', // Yellow for 4-6
                        '#0ea5e9', '#0ea5e9', '#0ea5e9', '#0ea5e9' // Cyan for 7-10
                    ],
                    borderRadius: 4,
                    borderWidth: 0
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        backgroundColor: '#0f172a',
                        titleColor: '#f8fafc',
                        bodyColor: '#cbd5e1',
                        borderColor: '#334155',
                        borderWidth: 1
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: '#1e293b' }},
                        ticks: {{ color: '#94a3b8', font: {{ family: 'JetBrains Mono' }} }}
                    }},
                    y: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#cbd5e1', font: {{ weight: 'bold' }} }}
                    }}
                }}
            }}
        }});

        // Paginated Filter Table Rows
        let currentPage = 1;
        const rowsPerPage = 10;

        function updateTable() {{
            const searchVal = document.getElementById('searchEmp').value.toLowerCase();
            const statusVal = document.getElementById('statusFilter').value;
            const rows = document.querySelectorAll('#employeeTableBody tr');
            
            // 1. Identify matching rows
            let matchingRows = [];
            rows.forEach(row => {{
                const empId = row.querySelector('.emp-id').innerText.toLowerCase();
                const isHighRisk = row.dataset.highRisk === 'true';
                
                const matchesSearch = empId.includes(searchVal);
                let matchesStatus = true;
                if (statusVal === 'high') {{
                    matchesStatus = isHighRisk;
                }} else if (statusVal === 'normal') {{
                    matchesStatus = !isHighRisk;
                }}
                
                if (matchesSearch && matchesStatus) {{
                    matchingRows.push(row);
                }} else {{
                    row.style.display = 'none';
                }}
            }});
            
            // 2. Paginate matching rows
            const totalMatching = matchingRows.length;
            const totalPages = Math.ceil(totalMatching / rowsPerPage) || 1;
            
            // Clamp currentPage
            if (currentPage > totalPages) currentPage = totalPages;
            if (currentPage < 1) currentPage = 1;
            
            const startIndex = (currentPage - 1) * rowsPerPage;
            const endIndex = Math.min(startIndex + rowsPerPage, totalMatching);
            
            // Show only the current page rows
            rows.forEach(row => row.style.display = 'none');
            for (let i = startIndex; i < endIndex; i++) {{
                matchingRows[i].style.display = '';
            }}
            
            // Update labels and buttons
            document.getElementById('paginationInfo').innerText = `Showing ${{totalMatching === 0 ? 0 : startIndex + 1}} to ${{endIndex}} of ${{totalMatching}} entries`;
            
            const prevBtn = document.getElementById('prevPageBtn');
            const nextBtn = document.getElementById('nextPageBtn');
            
            prevBtn.disabled = currentPage === 1;
            nextBtn.disabled = currentPage === totalPages;
            
            // Apply styling classes for disabled states
            if (currentPage === 1) {{
                prevBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }} else {{
                prevBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }}
            
            if (currentPage === totalPages) {{
                nextBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }} else {{
                nextBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }}
        }}

        function nextPage() {{
            currentPage++;
            updateTable();
        }}

        function prevPage() {{
            currentPage--;
            updateTable();
        }}

        function onFilterChange() {{
            currentPage = 1;
            updateTable();
        }}

        // Initialize on load
        window.addEventListener('DOMContentLoaded', () => {{
            updateTable();
        }});

    </script>
</body>
</html>
    """

    # Write HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"  [OK] HTML Report generated: {cf.style(output_path, cf.CYAN, cf.BOLD)}")
    
    # Automatically open in browser
    import webbrowser
    webbrowser.open('file:///' + os.path.abspath(output_path))



def generate_cox_html_report(
    report_df: pd.DataFrame,
    train_cindex: float,
    test_cindex: float,
    output_path: str = None
) -> None:
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, "report_cox.html")
    """
    Generate an interactive executive HTML dashboard for the Cox Proportional Hazards model.
    """
    # Group variables
    killers = report_df[(report_df['Significant'] == 'Yes') & (report_df['Direction'] == 'Increases Risk')]
    shields = report_df[(report_df['Significant'] == 'Yes') & (report_df['Direction'] == 'Decreases Risk')]
    noise = report_df[report_df['Significant'] == 'No']

    # Key takeaway recommendations
    income_words = ['income', 'salary', 'rate', 'daily', 'monthly', 'hourly']
    income_columns = [col for col in report_df.index if any(w in col.lower() for w in income_words)]
    
    recommendation_title = "RETENTION INSIGHT: CULTURAL FACTORS"
    recommendation_class = "border-sky-500/30 bg-sky-950/20 text-sky-200"
    recommendation_badge = "bg-sky-500/20 text-sky-400 border-sky-500/30"
    recommendation_text = (
        "Focus on non-monetary retention drivers (e.g., job role structure, manager quality, work-life balance) "
        "to reduce flight risks. Compensation indicators show minimal statistical correlation to attrition in this cohort."
    )
    
    if income_columns:
        any_sig = any(report_df.loc[col, 'Significant'] == 'Yes' for col in income_columns)
        if any_sig:
            recommendation_title = "CRITICAL COMPENSATION INSIGHT"
            recommendation_class = "border-rose-500/30 bg-rose-950/20 text-rose-200"
            recommendation_badge = "bg-rose-500/20 text-rose-400 border-rose-500/30"
            recommendation_text = (
                "Compensation actually drives attrition. Some salary-related factors are statistically significant "
                "in this cohort. Pay reviews and comp adjustments are highly recommended for vulnerable roles."
            )
        else:
            recommendation_title = "RETENTION INSIGHT: COMPENSATION IS NOT KEY"
            recommendation_class = "border-emerald-500/30 bg-emerald-950/20 text-emerald-200"
            recommendation_badge = "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
            recommendation_text = (
                "Money is NOT the main problem. Monthly income and salary hikes are statistically insignificant. "
                "Workplace culture, manager relationships, and overtime are the primary drivers of employee attrition."
            )

    # 1. Hidden Killers Table
    killers_list = []
    for name, row in killers.iterrows():
        clean_name = name.replace('_', ' ')
        if 'OverTime' in name and 'Yes' in name:
            clean_name = 'Working Overtime'
        elif 'MaritalStatus' in name:
            clean_name = clean_name.replace('MaritalStatus', 'Marital Status:')
        elif 'Department' in name:
            clean_name = clean_name.replace('Department', 'Department:')
        else:
            clean_name = clean_name.title()
        
        killers_list.append({
            "name": clean_name,
            "change": f"+{row['Risk_Change_%']}%",
            "p_val": f"{row['P_Value']:.3f}",
            "multiplier": f"{row['Multiplier']:.2f}x"
        })

    # 2. Retention Shields Table
    shields_list = []
    for name, row in shields.iterrows():
        clean_name = name.replace('_', ' ').title()
        impact = abs(row['Risk_Change_%'])
        
        shields_list.append({
            "name": clean_name,
            "change": f"-{impact}%",
            "p_val": f"{row['P_Value']:.3f}",
            "multiplier": f"{row['Multiplier']:.2f}x"
        })

    # 3. False Alarms List
    noise_list = []
    for name, row in noise.iterrows():
        clean_name = name.replace('_', ' ').title()
        noise_list.append({
            "name": clean_name,
            "p_val": f"{row['P_Value']:.3f}"
        })

    # Compile HTML
    html_content = HTML_HEAD.format(title="Cox Survival Risk Executive Summary")
    
    html_content += f"""
    <!-- Main Content Grid -->
    <main class="flex-1 overflow-y-auto p-6 md:p-8 space-y-6 max-w-7xl mx-auto w-full">
        
        <!-- Dashboard Header Title -->
        <div class="border-l-4 border-emerald-500 pl-4 py-2">
            <h1 class="text-3xl font-title font-bold tracking-tight text-white">Cox Survival Risk Executive Summary</h1>
            <p class="text-slate-400 mt-1">Statistical analysis representing the hazard ratio multipliers and significance levels of attrition factors.</p>
        </div>

        <!-- Row 1: KPI Stats Grid -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <p class="text-xs text-rose-400 font-medium uppercase tracking-wider">Hidden Attrition Killers</p>
                <h3 class="text-3xl font-mono-data font-bold mt-2 text-rose-500">{len(killers_list)}</h3>
                <p class="text-xs text-slate-500 mt-1">Significant risk multipliers</p>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <p class="text-xs text-emerald-400 font-medium uppercase tracking-wider">Active Retention Shields</p>
                <h3 class="text-3xl font-mono-data font-bold mt-2 text-emerald-500">{len(shields_list)}</h3>
                <p class="text-xs text-slate-500 mt-1">Significant protective factors</p>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <p class="text-xs text-emerald-400 font-medium uppercase tracking-wider">Generalization C-Index</p>
                <h3 class="text-3xl font-mono-data font-bold mt-2 text-emerald-500">{test_cindex * 100:.1f}%</h3>
                <p class="text-xs text-slate-500 mt-1">Accuracy on out-of-sample data</p>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
                <p class="text-xs text-yellow-400 font-medium uppercase tracking-wider">Statistical Noise</p>
                <h3 class="text-3xl font-mono-data font-bold mt-2 text-yellow-500">{len(noise_list)}</h3>
                <p class="text-xs text-slate-500 mt-1">Factors with P-Value &ge; 0.05</p>
            </div>
        </div>

        <!-- Row 2: Strategy Takeaway -->
        <div class="border rounded-xl p-6 {recommendation_class}">
            <div class="flex items-center space-x-3 mb-3">
                <span class="text-xs uppercase font-bold tracking-wider px-3 py-1 border rounded-full {recommendation_badge}">
                    {recommendation_title}
                </span>
            </div>
            <p class="text-sm font-medium leading-relaxed">{recommendation_text}</p>
        </div>

        <!-- Row 3: Two Columns for Killers & Shields -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            <!-- Hidden Killers Card -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
                <div class="p-5 border-b border-slate-800 bg-rose-950/10">
                    <h3 class="text-lg font-title font-bold text-rose-400 flex items-center gap-2">
                        <span>✖</span> The Hidden Attrition Killers
                    </h3>
                    <p class="text-xs text-slate-400 mt-1">Variables that significantly increase risk per unit change.</p>
                </div>
                <div class="overflow-x-auto flex-1">
                    <table class="w-full text-left text-sm">
                        <thead>
                            <tr class="border-b border-slate-800 bg-slate-950/20 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                <th class="py-2.5 px-4">Factor</th>
                                <th class="py-2.5 px-4 text-right">Risk Increase</th>
                                <th class="py-2.5 px-4 text-right">Multiplier</th>
                                <th class="py-2.5 px-4 text-right">P-Value</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/40 text-slate-300 font-medium">
    """
    
    for item in killers_list:
        html_content += f"""
                            <tr class="hover:bg-slate-800/20">
                                <td class="py-3 px-4 text-white font-semibold">{item["name"]}</td>
                                <td class="py-3 px-4 text-right text-rose-400 font-mono-data font-bold">{item["change"]}</td>
                                <td class="py-3 px-4 text-right font-mono-data">{item["multiplier"]}</td>
                                <td class="py-3 px-4 text-right font-mono-data text-slate-400">{item["p_val"]}</td>
                            </tr>
        """
        
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Retention Shields Card -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
                <div class="p-5 border-b border-slate-800 bg-emerald-950/10">
                    <h3 class="text-lg font-title font-bold text-emerald-400 flex items-center gap-2">
                        <span>✔</span> The Active Retention Shields
                    </h3>
                    <p class="text-xs text-slate-400 mt-1">Variables that significantly reduce risk per unit change.</p>
                </div>
                <div class="overflow-x-auto flex-1">
                    <table class="w-full text-left text-sm">
                        <thead>
                            <tr class="border-b border-slate-800 bg-slate-950/20 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                                <th class="py-2.5 px-4">Factor</th>
                                <th class="py-2.5 px-4 text-right">Risk Reduction</th>
                                <th class="py-2.5 px-4 text-right">Multiplier</th>
                                <th class="py-2.5 px-4 text-right">P-Value</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/40 text-slate-300 font-medium">
    """
    
    for item in shields_list:
        html_content += f"""
                            <tr class="hover:bg-slate-800/20">
                                <td class="py-3 px-4 text-white font-semibold">{item["name"]}</td>
                                <td class="py-3 px-4 text-right text-emerald-400 font-mono-data font-bold">{item["change"]}</td>
                                <td class="py-3 px-4 text-right font-mono-data">{item["multiplier"]}</td>
                                <td class="py-3 px-4 text-right font-mono-data text-slate-400">{item["p_val"]}</td>
                            </tr>
        """
        
    html_content += f"""
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

        <!-- Row 4: False Alarms List -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm">
            <h3 class="text-lg font-title font-bold text-white mb-2 flex items-center gap-2">
                <span class="text-yellow-500">⚠</span> The False Alarms (Insignificant Factors)
            </h3>
            <p class="text-xs text-slate-400 leading-relaxed mb-4">
                These variables failed to clear the significance boundary (P-Value &ge; 0.05). Corporate resources should not be allocated to modifying these parameters:
            </p>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
    """

    for item in noise_list:
        html_content += f"""
                <div class="bg-slate-950/50 p-3 rounded-lg border border-slate-800/60 flex items-center justify-between">
                    <span class="text-sm font-semibold text-slate-300 truncate mr-2" title="{item["name"]}">{item["name"]}</span>
                    <span class="text-xs font-mono-data text-slate-500">p={item["p_val"]}</span>
                </div>
        """

    html_content += """
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-600">
        &copy; 2026 MNC Enterprise Retention Systems. All Rights Reserved. Confidential - Internal Use Only.
    </footer>
</body>
</html>
    """

    # Write HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"  [OK] HTML Report generated: {cf.style(output_path, cf.CYAN, cf.BOLD)}")
    
    # Automatically open in browser
    import webbrowser
    webbrowser.open('file:///' + os.path.abspath(output_path))

