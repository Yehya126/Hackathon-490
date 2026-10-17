"""Self-contained HTML report: one tab per branch with tomorrow's expected sales, trends and alerts."""

from __future__ import annotations

import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FLAG_LABEL = {"possible_stockout": "Possible stockout", "demand_drop": "Drop", "demand_spike": "Spike",
              "sustained_drop": "Sustained drop", "sustained_rise": "Sustained rise"}

CSS = """
:root{--bg:#f7f6f3;--card:#fff;--ink:#1d1d1b;--muted:#6b6a65;--line:#e4e2dc;--accent:#2f6f5e;
--up:#2f6f5e;--down:#b4442f;--chip:#eef3f1}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171716;--card:#21211f;--ink:#ecebe7;
--muted:#a3a29c;--line:#34332f;--accent:#7cc4ad;--up:#7cc4ad;--down:#e58a74;--chip:#263330}}
:root[data-theme="dark"]{--bg:#171716;--card:#21211f;--ink:#ecebe7;--muted:#a3a29c;--line:#34332f;
--accent:#7cc4ad;--up:#7cc4ad;--down:#e58a74;--chip:#263330}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 10px}p.sub{color:var(--muted);margin:0 0 20px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
.tabs button{border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 16px;border-radius:999px;
cursor:pointer;font:inherit}.tabs button[aria-selected=true]{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:650}.kpi .l{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 12px;text-align:left;
border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:550}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:0}
.chip{background:var(--chip);border-radius:6px;padding:2px 8px;font-size:12px}
.up{color:var(--up)}.down{color:var(--down)}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:700px){.grid2{grid-template-columns:1fr}}
img{max-width:100%;display:block}.note{color:var(--muted);font-size:13px;margin-top:8px}
"""


def _chart(test: pd.DataFrame, branch: str) -> str:
    d = test[test["Branch"] == branch].groupby("Date")[["units", "forecast", "naive"]].sum()
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    ax.plot(d.index, d["units"], color="#6b6a65", lw=1.6, label="Actual units")
    ax.plot(d.index, d["forecast"], color="#2f6f5e", lw=2, label="Forecast")
    ax.plot(d.index, d["naive"], color="#c9a26b", lw=1, ls="--", label="Same weekday last week")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _table(df: pd.DataFrame, numeric: set[str]) -> str:
    head = "".join(f'<th class="{"n" if c in numeric else ""}">{html.escape(c)}</th>' for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f'<td class="{"n" if c in numeric else ""}">{v}</td>' for c, v in row.items()) + "</tr>"
        for _, row in df.iterrows())
    return f'<div class="card"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'


def _branch_section(branch, test, future, alerts, metrics, tomorrow, category_plan, fortnight, by_horizon) -> str:
    # Accuracy of the model that produces the forward forecasts (multi-day model, 1 day ahead, test period).
    day1 = by_horizon.loc[(branch, 1)]
    m_naive = metrics["Same weekday last week"].loc[branch]
    n_alerts = len(alerts[alerts["Branch"] == branch])
    days = test["Date"].nunique()
    kpis = "".join([
        _kpi(f"{day1['product_wape']:.0%}", f"Product-level forecast error (WAPE) vs {m_naive['WAPE']:.0%} for same-weekday-last-week"),
        _kpi(f"{day1['branch_day_wape']:.0%}", "Error on the branch's daily total"),
        _kpi(f"{day1['bias']:+.1%}", "Bias (forecast vs actual units sold)"),
        _kpi(f"{n_alerts}", f"Alerts over {days} test days"),
    ])

    f = future[future["Branch"] == branch].sort_values(["Category", "forecast"], ascending=[True, False])
    expected = pd.DataFrame({
        "Product": f["Product"].astype(str),
        "Category": f["Category"].astype(str),
        "Expected units": f["forecast"].map("<b>{:.1f}</b>".format),
        "Usual (last 4 same weekdays)": f["same_weekday_mean_4"].map("{:.1f}".format),
    })

    moves = f[f["vs_recent_same_weekday"].abs() >= 1].sort_values("vs_recent_same_weekday")
    def move_rows(rows, cls):
        return pd.DataFrame({
            "Product": rows["Product"].astype(str),
            "Tomorrow": rows["forecast"].map("{:.1f}".format),
            "Usual (last 4 same weekdays)": rows["same_weekday_mean_4"].map("{:.1f}".format),
            "Change": rows["vs_recent_same_weekday"].map(lambda v: f'<span class="{cls}">{v:+.1f}</span>'),
        })
    up = move_rows(moves.tail(6).iloc[::-1], "up")
    down = move_rows(moves.head(6), "down")

    a = alerts[alerts["Branch"] == branch].sort_values("severity", ascending=False).head(12)
    alert_table = pd.DataFrame({
        "Date": a["Date"].dt.strftime("%a %d %b"),
        "Type": a["flag"].map(lambda x: f'<span class="chip">{FLAG_LABEL[x]}</span>'),
        "Item": a["Product"].astype(str),
        "Actual": a["units"].map("{:.0f}".format),
        "Expected": a["expected"].map("{:.1f}".format),
        "Profit impact": a["profit_impact"].map(
            lambda v: f'<span class="{"up" if v > 0 else "down"}">{"+" if v > 0 else "−"}${abs(v):.0f}</span>'),
    })
    no_move = '<p class="note">No product is expected to move by a unit or more.</p>'
    move_cols = {"Tomorrow", "Usual (last 4 same weekdays)", "Change"}

    return f"""
<section data-branch="{branch}">
  <div class="kpis">{kpis}</div>
  <h2>Expected units sold on {tomorrow:%A %d %B}</h2>
  {_table(expected, {"Expected units", "Usual (last 4 same weekdays)"})}
  <h2>Expected units by category</h2>
  {_table(_category_table(category_plan, branch), {"Expected units"})}
  <div class="grid2">
    <div><h2>Expected higher than usual</h2>{_table(up, move_cols) if len(up) else no_move}</div>
    <div><h2>Expected lower than usual</h2>{_table(down, move_cols) if len(down) else no_move}</div>
  </div>
  {_fortnight_section(fortnight, by_horizon, branch)}
  <h2>Alerts worth checking ({n_alerts} over {days} test days)</h2>
  {_table(alert_table, {"Actual", "Expected", "Profit impact"}) if len(a) else '<p class="note">No alerts.</p>'}
  <h2>Daily units: actual vs forecast (test period)</h2>
  <div class="card" style="padding:12px"><img alt="Daily units actual vs forecast for {branch}" src="data:image/png;base64,{_chart(test, branch)}"></div>
</section>"""


def _fortnight_section(fortnight: pd.DataFrame, by_horizon: pd.DataFrame, branch: str) -> str:
    f = fortnight[fortnight["Branch"].astype(str) == branch]
    start = f["Date"].min()
    week_two = start + pd.Timedelta(days=7)

    days = f.groupby("Date")["forecast"].sum().reset_index()
    days["horizon"] = (days["Date"] - start).dt.days + 1
    errors = by_horizon.loc[branch, "branch_day_wape"]
    days["branch_day_wape"] = days["horizon"].map(errors)
    daily_table = pd.DataFrame({
        "Date": days["Date"].dt.strftime("%a %d %b"),
        "Days ahead": days["horizon"].astype(str),
        "Expected units (whole branch)": days["forecast"].map("<b>{:.0f}</b>".format),
        "Typical error on test": days["branch_day_wape"].map("±{:.0%}".format),
    })

    weeks = f.assign(week=(f["Date"] >= week_two).map({False: "week1", True: "week2"}))
    by_product = weeks.pivot_table(index=["Category", "Product"], columns="week", values="forecast",
                                   aggfunc="sum", observed=True).reset_index()
    by_product["total"] = by_product["week1"] + by_product["week2"]
    by_product = by_product.sort_values(["Category", "total"], ascending=[True, False])
    week1 = f"{start:%d %b}–{start + pd.Timedelta(days=6):%d %b}"
    week2 = f"{week_two:%d %b}–{week_two + pd.Timedelta(days=6):%d %b}"
    product_table = pd.DataFrame({
        "Product": by_product["Product"].astype(str),
        "Category": by_product["Category"].astype(str),
        week1: by_product["week1"].map("{:.0f}".format),
        week2: by_product["week2"].map("{:.0f}".format),
        "14 days": by_product["total"].map("<b>{:.0f}</b>".format),
    })
    return f"""<h2>Next 14 days: expected units per day</h2>
  {_table(daily_table, {"Days ahead", "Expected units (whole branch)", "Typical error on test"})}
  <p class="note">Typical error = how far this branch's daily total was off on average in the test weeks,
  for forecasts made that many days in advance.</p>
  <h2>Next 14 days: expected units per product</h2>
  {_table(product_table, {week1, week2, "14 days"})}"""


def _category_table(plan: pd.DataFrame, branch: str) -> pd.DataFrame:
    p = plan[plan["Branch"] == branch]
    p = pd.concat([p[p["Category"] != "ALL"], p[p["Category"] == "ALL"]])
    return pd.DataFrame({
        "Category": p["Category"].map(lambda c: "<b>Whole branch</b>" if c == "ALL" else html.escape(c)),
        "Expected units": p["forecast"].map("{:.0f}".format),
    })


def write_report(path, *, test, future, alerts, metrics, tomorrow, category_plan, fortnight, by_horizon) -> None:
    branches = sorted(test["Branch"].astype(str).unique())
    tabs = "".join(f'<button role="tab" data-b="{b}" aria-selected="{str(i == 0).lower()}">{b}</button>'
                   for i, b in enumerate(branches))
    sections = "".join(_branch_section(b, test, future, alerts, metrics, tomorrow, category_plan,
                                       fortnight, by_horizon) for b in branches)
    start, end = test["Date"].min(), test["Date"].max()
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Branch Sales Forecast</title>
<style>{CSS}</style></head><body><main>
<h1>Branch Sales Forecast</h1>
<p class="sub">Expected units sold per product for tomorrow and the next 14 days, plus unusual-sales alerts — each branch judged against its
own history. Evaluated on {start:%d %b}–{end:%d %b %Y}; forecast from {tomorrow:%d %b %Y}.</p>
<div class="tabs" role="tablist">{tabs}</div>
{sections}
</main><script>
const tabs=[...document.querySelectorAll('.tabs button')],secs=[...document.querySelectorAll('section')];
function show(b){{tabs.forEach(t=>t.setAttribute('aria-selected',t.dataset.b===b));
secs.forEach(s=>s.hidden=s.dataset.branch!==b);}}
tabs.forEach(t=>t.onclick=()=>show(t.dataset.b));show(tabs[0].dataset.b);
</script></body></html>"""
    path.write_text(page, encoding="utf-8")
