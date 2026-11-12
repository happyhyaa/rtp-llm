"""Render verified compare() rows as a standalone HTML report."""
from html import escape


def render(rows):
    def pct(value):
        return "—" if value is None else f"{100 * value:.5f}%"
    def table(headings, records):
        head = "".join(f"<th>{escape(h)}</th>" for h in headings)
        body = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>" for values in records)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    summary = table(["组别", "FULL 命中率", "联合命中率", "距 FULL 少命中 tokens", "相对首组达标率"],
                    [(r["run"], pct(r["full_rate"]), pct(r["joint_rate"]), f"{r['lost_to_joint_tokens']:,}",
                      pct(r["attainment"])) for r in rows])
    details = []
    for r in rows:
        metrics = table(["回收计数器", "累计值"], sorted(r["evictions"].items()))
        details.append(f'<details><summary>{escape(r["run"])} · 后端与回收明细</summary>'
                       f'<p>{escape(r["revision"])}</p>{metrics}</details>')
        if "last_stage_cases" in r:
            c = r["last_stage_cases"]
            details.append(f'<details><summary>{escape(r["run"])} · 逐请求差异</summary>'
                           f'<p>本阶段 {c["requests"]:,} 条，变长 {c["improved"]:,} 条，变短 {c["regressed"]:,} 条。</p>'
                           + table(["请求序号","少命中 tokens"], [(x["request"], x["loss"]) for x in c["worst"]])
                           + '</details>')
    first = rows[0]
    return ("""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>BTC 离线命中率对照</title><style>body{max-width:1100px;margin:4vw auto;padding:0 1rem;color:#183344;font:16px/1.55 system-ui}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:.6rem;text-align:right;border-bottom:1px solid #ccd8db}
th:first-child,td:first-child{text-align:left}th{background:#e8f2ef}details{margin:1rem 0}summary{cursor:pointer}</style>
<h1>BTC 离线命中率对照</h1>"""
            + f'<p>预热 {first["warmup"]:,} 条，正式统计 {first["requests"]:,} 条，共 {first["input_tokens"]:,} 个输入 tokens。</p>'
            + summary + "".join(details) + "</html>")

def with_charts(summary, runs):
    """Place the detailed report bodies in the landing page and rebase their local links."""
    import re
    import posixpath
    from urllib.parse import quote
    groups = []
    for i, (label, directory) in enumerate(runs, 1):
        detail = (directory / "report/report.html").read_text(encoding="utf-8")
        if i == 1:
            css = detail.split("<style>", 1)[1].split("</style>", 1)[0]
        body = detail.split("<main>", 1)[1].split("</main>", 1)[0]
        body = body.split("</h1>", 1)[1]
        prefix = f"run-{i}-"
        def link(match):
            attr, value = match.groups()
            if attr == "id":
                value = prefix + value
            elif value.startswith("#"):
                value = "#" + prefix + value[1:]
            else:
                value = posixpath.normpath(posixpath.join(quote(directory.name), "report", value))
            return f'{attr}="{value}"'
        body = re.sub(r'\b(id|href|src)="([^"]*)"', link, body)
        name = {"continuous":"连续回放","resumed":"断点续跑"}.get(label.split("/")[-1], label)
        if i == 1:
            groups.append(f"<h2>{escape(name)}</h2>" + body)
        else:
            groups.append(f"<details><summary>{escape(name)} · 完整分析图表</summary>{body}</details>")
    start = summary.index("<h1>")
    intro = summary[start:summary.index("</html>")]
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>BTC 离线命中率评估</title><style>' + css +
            '</style><main>' + intro + ''.join(groups) + '</main></html>')
