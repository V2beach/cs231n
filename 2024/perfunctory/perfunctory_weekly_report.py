import os
from datetime import datetime, timedelta

def get_week_number(date):
    """返回给定日期的周数"""
    year, week, _ = date.isocalendar()
    return year, week

def split_markdown_by_weeks(input_file, output_dir, start_date):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取原始 markdown 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按照 '## ' 来分割内容，分割后每个部分将是一个段落
    sections = content.split('\n## ')

    # 处理第一个部分（包括第一个 '##'，直接保存）
    first_section = (sections[0] + sections[1]).strip()  # 第一部分不需要进一步处理

    # 获取当前周次和日期
    current_date = start_date
    year, week = get_week_number(current_date)

    # 生成文件名，例如 "2024年第1周.md"
    output_file = os.path.join(output_dir, f"{year}年第{week}周.md")

    # 写入第一个文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"## {first_section}")  # 给第一个部分加上标题
    print(f"文件 {output_file} 已保存。")

    # 处理从第二个 '##' 开始的部分
    for i, section in enumerate(sections[1:], 1):  # 从第二个部分开始
        section = section.strip()  # 去掉前后的空白字符
        section = '## ' + section  # 给每个部分加上 '## '，表示二级标题

        # 获取当前周次和日期
        current_date = start_date - timedelta(weeks=i)  # 向前推移 i 周
        year, week = get_week_number(current_date)

        # 生成文件名
        output_file = os.path.join(output_dir, f"{year}年第{week}周.md")

        # 写入新的文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(section)

        print(f"文件 {output_file} 已保存。")

# 使用示例
input_file = 'report.md'  # 需要分割的 markdown 文件
output_dir = 'reports_by_weeks'  # 保存分割后的文件的目录
start_date = datetime(2024, 12, 12)  # 开始日期，今天

split_markdown_by_weeks(input_file, output_dir, start_date)

# pypandoc markdown to pdf
import pypandoc 
# pypandoc.download_pandoc()  # 直接在當前路徑下載 pandoc-3.6-x86_64-macOS.pkg

def markdown_to_pdf(md_file, pdf_file):
    # 使用 Pandoc 将 Markdown 转换为 PDF
    output = pypandoc.convert_file(md_file, 'pdf', outputfile=pdf_file, extra_args=['--pdf-engine=pdflatex'])  # install basictex & tlmgr install bookmark (只是一個 bookmark.sty 文件)
    print(f"PDF 文件已保存到 {pdf_file}")

# tlmgr update --self --all
# pandoc -s cs231n.md -o output.pdf --verbose --pdf-engine=pdflatex/lualatex/xelatex 轉不了，太垃圾