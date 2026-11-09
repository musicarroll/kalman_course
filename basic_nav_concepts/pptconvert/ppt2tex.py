from pptx import Presentation
import re

def clean_text(text):
    # Define a dictionary of Greek and math symbol replacements
    greek_replacements = {
        "Δ": r"$\Delta$",
        "α": r"$\alpha$",
        "β": r"$\beta$",
        # Add more symbols as needed
    }
    for char, latex_eq in greek_replacements.items():
        text = text.replace(char, latex_eq)
    
    # Escape LaTeX special characters outside of list items
    text = re.sub(r"(?<!\\)&", r"\\&", text)
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    text = re.sub(r"(?<!\\)#", r"\\#", text)
    text = re.sub(r"(?<!\\)_", r"\\_", text)
    text = re.sub(r"(?<!\\){", r"\\{", text)
    text = re.sub(r"(?<!\\)}", r"\\}", text)

    # Ensure only standalone math expressions are wrapped in $...$
    math_patterns = [
        r'\b\d+(\.\d+)?\b',  # Matches standalone numbers
        r'\b[a-zA-Z]+\b',     # Matches standalone variables or symbols
        r'[\+\-\*/=<>]',      # Matches operators
        r'\b(sin|cos|tan|log|ln)\b'  # Matches functions
    ]
    
    # Wrap patterns in $...$
    for pattern in math_patterns:
        text = re.sub(pattern, lambda match: f"${match.group(0)}$", text)
    
    # Additional Unicode replacements
    text = text.replace("\u000B", " ")  # Replace U+000B (Vertical Tab) with space

    return text

def ppt_to_beamer(pptx_path, tex_path):
    # Load the PowerPoint presentation
    prs = Presentation(pptx_path)
    
    # Start building the LaTeX Beamer content
    beamer_content = [
        r"\documentclass{beamer}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath, amssymb, amsthm}",
        r"\title{Your Presentation Title}",
        r"\author{Your Name}",
        r"\date{\today}",
        r"\begin{document}",
        r"\frame{\titlepage}"
    ]
    
    # Iterate over slides
    for slide_num, slide in enumerate(prs.slides, start=1):
        frame_content = [r"\begin{frame}"]

        # Extract title and subtitle if they exist and are text placeholders
        title = clean_text(slide.shapes.title.text) if slide.shapes.title and hasattr(slide.shapes.title, "text") else ""
        subtitle = ""
        
        # Try to extract subtitle from placeholders, if available
        for shape in slide.placeholders:
            if shape != slide.shapes.title and hasattr(shape, "text"):
                subtitle = clean_text(shape.text)
                break
        
        # Insert the frame title and subtitle
        frame_content.append(rf"\onslide<1->{{\frametitle{{{title}}} \framesubtitle{{{subtitle}}}}}")
        frame_content.append(r"\begin{itemize}")

        # Counter to keep track of the current onscreen index
        onscreen_index = 2

        # Extract text from each shape and structure it into nested bullet points
        for shape in slide.shapes:
            if shape.has_text_frame:
                # Main bullet points
                for paragraph in shape.text_frame.paragraphs:
                    frame_content.append(rf"\onslide<{onscreen_index}->{{\item {clean_text(paragraph.text)}}}")
                    onscreen_index += 1

                    # Sub-bullet points
                    if paragraph.level > 0:
                        frame_content.append(r"\begin{itemize}")
                        for run in paragraph.runs:
                            frame_content.append(rf"\onslide<{onscreen_index}->{{\item {clean_text(run.text)}}}")
                            onscreen_index += 1
                        frame_content.append(r"\end{itemize}")

        frame_content.append(r"\end{itemize}")
        frame_content.append(r"\onslide<{}->{{\center --------------------------------------------------------------------}}".format(onscreen_index))
        frame_content.append(r"\end{frame}")

        # Append the frame to the main content
        beamer_content.extend(frame_content)

    # End the document
    beamer_content.append(r"\end{document}")

    # Write to the output .tex file
    with open(tex_path, "w") as tex_file:
        tex_file.write("\n".join(beamer_content))
    
    print(f"LaTeX Beamer presentation created at {tex_path}")

# Usage
ppt_to_beamer("basic_nav_concepts_part_4.pptx", "basic_nav_concepts_part_4.tex")
