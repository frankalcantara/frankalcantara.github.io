#!/usr/bin/env python3
"""
Jekyll to Quarto Converter for Transformers Book
Converts Jekyll posts to Quarto format (.qmd) for the transformers book chapters.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

class JekyllToQuartoConverter:
    def __init__(self, source_dir: str, target_dir: str):
        """
        Initialize the converter with source and target directories.
        
        Args:
            source_dir: Path to frankalcantara.github.io/_posts
            target_dir: Path to frankalcantara.github.io/transformers-book/chapters
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Chapter mapping from post names to chapter files
        self.chapter_mapping = {
            'transformers-um.md': '01-fundamentos-matematicos.qmd',
            'transformers-dois.md': '02-representacao-linguagem.qmd', 
            'transformers-tres.md': '03-redes-neurais.qmd',
            'transformers-quatro.md': '04-word-embeddings.qmd',
            'transformers-cinco.md': '05-arquitetura-transformer.qmd',
            'transformers-seis.md': '06-mecanismos-atencao.qmd',
            'transformers-sete.md': '07-implementacoes-avancadas.qmd'
        }
        
        # Quarto headers for each chapter
        self.quarto_headers = {
            '01-fundamentos-matematicos.qmd': {
                'title': 'Fundamentos Matemáticos',
                'subtitle': 'Álgebra Linear e Cálculo para Transformers'
            },
            '02-representacao-linguagem.qmd': {
                'title': 'Representação de Linguagem',
                'subtitle': 'Tokenização e Codificação Textual'
            },
            '03-redes-neurais.qmd': {
                'title': 'Redes Neurais',
                'subtitle': 'Fundamentos para Arquiteturas Modernas'
            },
            '04-word-embeddings.qmd': {
                'title': 'Word Embeddings',
                'subtitle': 'Representações Vetoriais de Palavras'
            },
            '05-arquitetura-transformer.qmd': {
                'title': 'Arquitetura Transformer',
                'subtitle': 'Estrutura e Componentes Principais'
            },
            '06-mecanismos-atencao.qmd': {
                'title': 'Mecanismos de Atenção',
                'subtitle': 'Self-Attention e Multi-Head Attention'
            },
            '07-implementacoes-avancadas.qmd': {
                'title': 'Implementações Avançadas',
                'subtitle': 'Otimizações e Técnicas Práticas'
            }
        }

    def create_target_directory(self) -> None:
        """Create target directory if it doesn't exist."""
        self.target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Target directory created/verified: {self.target_dir}")

    def extract_front_matter(self, content: str) -> Tuple[Optional[Dict], str]:
        """
        Extract and remove Jekyll front matter from content.
        
        Args:
            content: Full markdown content with potential front matter
            
        Returns:
            Tuple of (front_matter_dict, content_without_front_matter)
        """
        front_matter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(front_matter_pattern, content, re.DOTALL)
        
        if match:
            try:
                front_matter = yaml.safe_load(match.group(1))
                content_without_fm = content[match.end():]
                return front_matter, content_without_fm
            except yaml.YAMLError:
                print("Warning: Could not parse front matter as YAML")
                return None, content
        
        return None, content

    def remove_blog_links(self, content: str) -> str:
        """
        Remove internal blog links and references.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with blog links removed
        """
        # Remove Jekyll site variables
        content = re.sub(r'\{\{\s*site\.url\s*\}\}', '', content)
        content = re.sub(r'\{\{\s*site\.baseurl\s*\}\}', '', content)
        
        # Remove internal blog post links
        content = re.sub(r'\[([^\]]+)\]\(/[^)]*\)', r'\1', content)
        
        # Remove Jekyll liquid tags
        content = re.sub(r'\{\%[^%]*\%\}', '', content)
        
        return content

    def remove_glossaries(self, content: str) -> str:
        """
        Remove glossary sections and references.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with glossaries removed
        """
        # Remove glossary sections (usually at the end)
        content = re.sub(r'\n## Glossário.*$', '', content, flags=re.DOTALL)
        content = re.sub(r'\n### Glossário.*$', '', content, flags=re.DOTALL)
        
        # Remove glossary references
        content = re.sub(r'\*\[([^\]]+)\]:\s*[^\n]*\n', '', content)
        
        return content

    def validate_latex(self, content: str) -> List[str]:
        """
        Validate LaTeX formatting in content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check for proper block math delimiters
        if '\\[' in content or '\\]' in content:
            warnings.append("Found \\[ or \\] - should use $$ for block math")
        
        # Check for unmatched $ delimiters
        dollar_count = content.count('$')
        if dollar_count % 2 != 0:
            warnings.append("Unmatched $ delimiters found")
        
        # Check for double backslashes in math mode (common error)
        math_blocks = re.findall(r'\$\$.*?\$\$', content, re.DOTALL)
        for block in math_blocks:
            if '\\\\' in block and not re.search(r'\\begin\{.*matrix.*\}', block):
                warnings.append("Found \\\\\\\\ in math block - may need adjustment")
        
        return warnings

    def validate_cpp_code(self, content: str) -> List[str]:
        """
        Validate C++ code blocks formatting.
        
        Args:
            content: Markdown content
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Find all C++ code blocks
        cpp_blocks = re.findall(r'```cpp\n(.*?)\n```', content, re.DOTALL)
        cpp_blocks.extend(re.findall(r'```c\+\+\n(.*?)\n```', content, re.DOTALL))
        
        for i, block in enumerate(cpp_blocks):
            # Check for basic C++ syntax elements
            if '#include' not in block and 'std::' not in block and 'namespace' not in block:
                if len(block.strip()) > 50:  # Only warn for substantial blocks
                    warnings.append(f"C++ block {i+1} may be missing includes or std:: qualifiers")
        
        return warnings

    def create_quarto_header(self, chapter_file: str, original_front_matter: Optional[Dict] = None) -> str:
        """
        Create Quarto YAML header for the chapter.
        
        Args:
            chapter_file: Target chapter filename
            original_front_matter: Original Jekyll front matter
            
        Returns:
            Formatted Quarto YAML header
        """
        if chapter_file not in self.quarto_headers:
            raise ValueError(f"No header template for {chapter_file}")
        
        header_info = self.quarto_headers[chapter_file]
        
        # Extract chapter number from filename
        chapter_num = chapter_file.split('-')[0]
        
        header = f"""---
title: "{header_info['title']}"
subtitle: "{header_info['subtitle']}"
chapter: {chapter_num}
format:
  html:
    code-fold: false
    code-tools: true
    toc: true
    toc-depth: 3
  pdf:
    documentclass: book
    geometry:
      - margin=1in
    include-in-header:
      - text: |
          \\usepackage{{amsmath,amssymb,amsfonts}}
          \\usepackage{{algorithm2e}}
          \\usepackage{{listings}}
bibliography: ../references.bib
---

"""
        return header

    def convert_references(self, content: str) -> str:
        """
        Convert references to Quarto format.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with converted references
        """
        # Convert basic citations
        content = re.sub(r'\[@([^\]]+)\]', r'[@\1]', content)
        
        # Convert figure references
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'![\\1](\\2)', content)
        
        return content

    def fix_image_paths(self, content: str) -> str:
        """
        Fix image paths for the new directory structure.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with corrected image paths
        """
        # Convert relative paths from _posts to chapters
        content = re.sub(r'!\[([^\]]*)\]\(\.\./images/', r'![\\1](../images/', content)
        content = re.sub(r'!\[([^\]]*)\]\(images/', r'![\\1](../images/', content)
        
        return content

    def process_single_file(self, source_file: Path, target_file: Path) -> bool:
        """
        Process a single Jekyll post file to Quarto format.
        
        Args:
            source_file: Source .md file path
            target_file: Target .qmd file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read source file
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"Processing: {source_file.name} -> {target_file.name}")
            
            # Extract and remove front matter
            front_matter, content = self.extract_front_matter(content)
            
            # Remove blog-specific elements
            content = self.remove_blog_links(content)
            content = self.remove_glossaries(content)
            
            # Fix paths and references
            content = self.fix_image_paths(content)
            content = self.convert_references(content)
            
            # Create Quarto header
            quarto_header = self.create_quarto_header(target_file.name, front_matter)
            
            # Combine header and content
            final_content = quarto_header + content
            
            # Write target file
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            # Perform validations
            latex_warnings = self.validate_latex(final_content)
            cpp_warnings = self.validate_cpp_code(final_content)
            
            if latex_warnings:
                print(f"  LaTeX warnings for {target_file.name}:")
                for warning in latex_warnings:
                    print(f"    - {warning}")
            
            if cpp_warnings:
                print(f"  C++ warnings for {target_file.name}:")
                for warning in cpp_warnings:
                    print(f"    - {warning}")
            
            if not latex_warnings and not cpp_warnings:
                print(f"  ✅ {target_file.name} processed successfully")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {source_file.name}: {e}")
            return False

    def convert_all_chapters(self) -> None:
        """Convert all mapped chapter files."""
        self.create_target_directory()
        
        successful_conversions = 0
        total_chapters = len(self.chapter_mapping)
        
        print(f"Starting conversion of {total_chapters} chapters...")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        print("-" * 60)
        
        for source_name, target_name in self.chapter_mapping.items():
            source_file = self.source_dir / source_name
            target_file = self.target_dir / target_name
            
            if not source_file.exists():
                print(f"❌ Source file not found: {source_file}")
                continue
            
            if self.process_single_file(source_file, target_file):
                successful_conversions += 1
        
        print("-" * 60)
        print(f"Conversion complete: {successful_conversions}/{total_chapters} chapters processed")
        
        if successful_conversions == total_chapters:
            print("🎉 All chapters converted successfully!")
        else:
            print(f"⚠️  {total_chapters - successful_conversions} chapters had issues")

def main():
    """Main function to run the converter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Jekyll posts to Quarto chapters')
    parser.add_argument('--source', '-s', 
                       default='frankalcantara.github.io/_posts',
                       help='Source directory with Jekyll posts')
    parser.add_argument('--target', '-t',
                       default='frankalcantara.github.io/transformers-book/chapters', 
                       help='Target directory for Quarto chapters')
    
    args = parser.parse_args()
    
    # Validate directories
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        return 1
    
    # Create converter and run
    converter = JekyllToQuartoConverter(args.source, args.target)
    converter.convert_all_chapters()
    
    return 0

if __name__ == '__main__':
    exit(main())