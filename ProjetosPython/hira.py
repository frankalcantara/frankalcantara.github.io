import re

def process_markdown_headers(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex para encontrar headers Markdown
    header_pattern = re.compile(r'^(#+)\s+(.*?)$', re.MULTILINE)
    headers = header_pattern.findall(content)

    current_level = 0
    processed_headers = []
    issues = []

    for line_num, (hashes, title) in enumerate(headers, 1):
        level = len(hashes)
        
        if level > current_level + 1:
            issues.append(f"Linha {line_num}: O título '{title}' (H{level}) está fora de ordem. " 
                          f"Esperado no máximo H{current_level + 1}.")
        elif level == 1 and current_level != 0:
            issues.append(f"Linha {line_num}: Múltiplos títulos H1 encontrados.")
        
        current_level = level
        processed_headers.append(f"{'#' * level} {title} (H{level})")

    # Escrever os headers processados no arquivo de saída
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(processed_headers))

    return issues

# Uso do script
if __name__ == "__main__":
    input_file_path = "teste.md"  # Substitua pelo caminho do seu arquivo de entrada
    output_file_path = "headers_extraidos.md"  # Arquivo de saída com os headers extraídos
    
    problems = process_markdown_headers(input_file_path, output_file_path)
    
    if problems:
        print("Problemas encontrados na hierarquia de títulos:")
        for problem in problems:
            print(problem)
    else:
        print("Nenhum problema encontrado na hierarquia de títulos.")
    
    print(f"\nOs headers foram extraídos e salvos em '{output_file_path}'.")