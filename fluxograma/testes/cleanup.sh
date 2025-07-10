#!/bin/bash
# Script de limpeza para remover arquivos desnecessários
# Execute este script se não precisar mais dos backups

echo "🧹 Limpeza de Arquivos - Fluxograma Interativo"
echo "=============================================="

# Lista de arquivos a serem removidos
files_to_remove=(
    "backup_fixed-parser.js"
    "backup_flowchart-parser.js"
    "backup_quick-fix.js"
    "backup_simple-parser.js"
    "backup_step-executor.js"
    "todo.md"
)

echo "📋 Arquivos que serão removidos:"
for file in "${files_to_remove[@]}"; do
    if [ -f "$file" ]; then
        echo "  ❌ $file"
    fi
done

echo ""
read -p "🤔 Deseja continuar? (s/N): " confirm

if [[ $confirm =~ ^[Ss]$ ]]; then
    echo ""
    echo "🗑️ Removendo arquivos..."
    
    for file in "${files_to_remove[@]}"; do
        if [ -f "$file" ]; then
            rm "$file"
            echo "  ✅ Removido: $file"
        else
            echo "  ⚠️ Não encontrado: $file"
        fi
    done
    
    echo ""
    echo "🎉 Limpeza concluída!"
    echo "📁 Estrutura final do projeto:"
    ls -la *.html *.css *.js *.md 2>/dev/null | grep -v backup
    
else
    echo "❌ Limpeza cancelada"
fi

echo ""
echo "🚀 Para usar o aplicativo, abra 'index.html' no navegador"
