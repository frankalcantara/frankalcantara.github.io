import sys
import os
import re
from datetime import datetime
import yaml
from urllib.parse import quote

def main(input_filename):
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extrair o Front Matter
    front_matter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if front_matter_match:
        front_matter_raw = front_matter_match.group(1)
        front_matter = yaml.safe_load(front_matter_raw)
        content = content[front_matter_match.end():]  # Conteúdo restante após o Front Matter
    else:
        print("Front Matter não encontrado no arquivo.")
        sys.exit(1)

    # Encontrar todos os títulos de nível 2 e suas posições
    pattern = re.compile(r'^# .*$\n?', re.MULTILINE)
    matches = list(pattern.finditer(content))

    # Preparar a lista para conter o conteúdo de cada seção
    sections = []
    for i in range(len(matches)):
        start = matches[i].start()
        if i + 1 < len(matches):
            end = matches[i+1].start()
        else:
            end = len(content)
        section_content = content[start:end]
        sections.append(section_content)

    # Criar os novos arquivos sem ajustar os níveis dos títulos
    date_str = datetime.now().strftime('%Y-%m-%d')
    num_sections = len(sections)
    filenames = []
    title_to_filename = {}  # Mapeamento de títulos para nomes de arquivos
    for idx, section in enumerate(sections):
        # Usar a seção como está, sem ajustar os títulos
        adjusted_section = section

        # Extrair o título a partir do heading
        title_match = re.match(r'^# (.*)', section)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = f"Seção {idx+1}"

        # Modificar o Front Matter
        new_front_matter = front_matter.copy()
        new_front_matter['title'] = title
        # Manter os campos existentes que não precisam ser alterados
        if 'tags' not in new_front_matter:
            new_front_matter['tags'] = []

        # Gerar 'beforetoc' com links
        beforetoc = ''
        if idx > 0:
            prev_title = get_title_from_section(sections[idx - 1])
            prev_filename = generate_filename(date_str, idx, prev_title)
            beforetoc += f"[Anterior]({quote(prev_filename)})\n"
        if idx < num_sections -1:
            next_title = get_title_from_section(sections[idx + 1])
            next_filename = generate_filename(date_str, idx + 2, next_title)  # idx + 2 porque idx começa em 0
            beforetoc += f"[Próximo]({quote(next_filename)})\n"
        new_front_matter['beforetoc'] = beforetoc.strip()

        # Preparar o conteúdo para escrever
        front_matter_str = '---\n' + yaml.dump(new_front_matter, allow_unicode=True) + '---\n'
        output_content = front_matter_str + adjusted_section

        # Gerar o nome do arquivo com o número da seção
        section_number = idx + 1  # idx começa em 0, então adicionamos 1
        filename = generate_filename(date_str, section_number, title)
        filenames.append(filename)  # Armazenar o nome do arquivo para uso posterior
        title_to_filename[title] = filename  # Mapeia o título para o nome do arquivo

        # Escrever no novo arquivo
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"Criado arquivo: {filename}")

    # Criar o arquivo de TOC
    create_toc_file(content, front_matter, date_str, num_sections + 1, title_to_filename)

def get_title_from_section(section):
    title_match = re.match(r'^## (.*)', section)
    if title_match:
        return title_match.group(1).strip()
    else:
        return "Sem Título"

def generate_filename(date_str, section_number, title):
    # Sanitizar o título para criar um nome de arquivo válido
    valid_title = re.sub(r'[\\/*?:"<>|]', '', title).replace(' ', '-')
    filename = f"{date_str}-{section_number}-{valid_title}.md"
    return filename

def create_toc_file(content, front_matter, date_str, section_number, title_to_filename):
    # Encontrar todos os títulos até o nível 4
    toc_pattern = re.compile(r'^(#{1,4})\s+(.*)', re.MULTILINE)
    toc_matches = toc_pattern.findall(content)

    # Construir o TOC com indentação apropriada
    toc_lines = []
    current_file = None
    for hashes, title in toc_matches:
        level = len(hashes)
        indent = '  ' * (level - 1)

        if level == 2:
            # Atualizar o arquivo atual
            current_file = title_to_filename.get(title, '')
            link = current_file
        else:
            # Criar âncora para títulos de nível 3 e 4
            anchor = re.sub(r'[^a-zA-Z0-9_-]', '', title.replace(' ', '-')).lower()
            if current_file:
                link = f"{current_file}#{anchor}"
            else:
                link = f"#{anchor}"

        toc_lines.append(f"{indent}- [{title}]({quote(link)})")

    toc_content = '\n'.join(toc_lines)

    # Modificar o Front Matter
    toc_front_matter = front_matter.copy()
    toc_front_matter['title'] = 'Índice'
    if 'tags' not in toc_front_matter:
        toc_front_matter['tags'] = []
    toc_front_matter['beforetoc'] = ''  # Pode adicionar links aqui se desejar

    # Preparar o conteúdo para escrever
    front_matter_str = '---\n' + yaml.dump(toc_front_matter, allow_unicode=True) + '---\n'
    output_content = front_matter_str + toc_content

    # Gerar o nome do arquivo
    filename = generate_filename(date_str, section_number, 'Indice')

    # Escrever no novo arquivo
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output_content)
    print(f"Criado arquivo de TOC: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py arquivo_entrada.md")
        sys.exit(1)
    input_filename = sys.argv[1]
    main(input_filename)
