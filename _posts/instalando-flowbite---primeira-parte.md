---
layout: ""
title: Instalando Flowbite - Primeira Parte
author: Frank
categories:
    - artigo
tags:
    - resolução de problemas
    - servidor
    - svelte
image: ""
featured: false
rating: -1
description: ""
date: 2025-01-26T15:58:07.375Z
preview: ""
keywords: ""
toc: true
published: false
beforetoc: ""
lastmod: 2025-01-26T15:59:37.580Z
slug: instalando-flowbite-primeira-parte
---


Depois de 10 anos com as mãos longe dos terminais linux e do desenvolvimento web resolvi criar um aplicativo web no tempo livre e, graças às vozes na minha cabeça escolhi o [Svelte](https://svelte.dev/), [Sveltekit](https://svelte.dev/tutorial/kit/introducing-sveltekit) e o [Flowbite](https://flowbite-svelte-admin-dashboard.vercel.app/) para criar a estrutura básica do projeto. Devia ter optado pelo C++.

A máquina que irá suportar esse ambiente é uma droplet da [Digital Ocean](https://cloud.digitalocean.com/) rodando Ubuntu 24.10 (GNU/Linux 6.11.0-13-generic x86_64) e um Nginx 1.26.0. Além de uma duzia, ou mais, de ambientes virtuais Python 3.12.7 e outros tantos ambientes configurados para C++ 17, 20 e 23. Tudo para testes e pesquisas.

Não foi fácil. Este texto apresenta a primeira parte do processo de instalação do SvelteKit com Tailwind CSS e Flowbite em um sistema Ubuntu limpo. No que se refere ao Nodejs e todos os ambientes Javascript/Typescript modernos.

## Preparação do Ambiente

Primeiro, precisamos nos certifica que o ambiente esteja atualizado e instalar as ferramentas essenciais:

```shell
# Atualiza a lista de pacotes e o sistema
sudo apt update
sudo apt upgrade -y

# Instala ferramentas essenciais de desenvolvimento
sudo apt install -y build-essential python3 curl git
```

## Instalação do Node.js

Vamos instalar o Node.js LTS (Long Term Support) através do repositório oficial NodeSource:

```shell
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 curl git
```

Instalando o [nodejs](https://nodejs.org/pt). Aqui foi necessário instala uma versão anterior, a versão 20.x porque encontrei alguns pacotes incompatíveis com o node mais recente 22.x.

```shell
# Adiciona o repositório NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -

# Instala Node.js e npm
sudo apt install -y nodejs

# Verifica as instalações
node --version   # Deve mostrar v20.x.x
npm --version    # Deve mostrar v9.x.x ou mais recente
```

## Instalação do Gerenciador de Pacotes 

Duas opções óbvias: `npm` e `pnpm`. 

O `npm` e `pnpm` são gerenciadores de pacotes para o ecossistema JavaScript, com algumas diferenças em termos de funcionamento e eficiência: 

O `npm` e o `pnpm` são gerenciadores de pacotes que diferem principalmente na forma como armazenam e gerenciam as dependências. O `npm` instala cada pacote diretamente na pasta _node_modules_ do projeto, o que pode levar à duplicação e consumo excessivo de espaço em disco. Já o `pnpm` utiliza um armazenamento centralizado e links simbólicos, evitando redundâncias e melhorando o desempenho, especialmente em projetos grandes.

O `pnpm` impõe uma estrutura mais organizada e estrita na pasta _node_modules_, garantindo que cada pacote acesse apenas suas dependências diretas, o que aumenta a consistência e reduz conflitos de versão. Embora o `npm` seja o gerenciador padrão do _Node.js_ e apresente maior compatibilidade com ferramentas existentes, o `pnpm` oferece vantagens significativas em termos de eficiência de armazenamento, velocidade e gerenciamento de dependências. Mas, é necessário atenção.

Escolhi o `pnpm` porque o guia de instalação do Flowbite sugere seu uso e porque eu tentei duas vezes com o `npm` e não consegui.

```shell
# Instala o pnpm globalmente
sudo npm install -g pnpm

# Verifica a instalação
pnpm --version  # Deve mostrar v9.x.x ou mais recente
```

## Criação do Projeto SvelteKit

Metade dos guias de instalação disponíveis na internet não funcionam por alguma incompatibilidade com algum dos módulos necessários para o Flowbite ou para o Tailwind então fui forçado a aprender como configurar o diretório do projeto:

```shell
# Cria o diretório do projeto
sudo mkdir -p /var/www/seu-projeto
sudo chown -R $USER:$USER /var/www/seu-projeto
cd /var/www/seu-projeto
```

Se funcionou, podemo criar o projeto SvelteKit e instalar suas dependências:

```shell
# Cria um novo projeto SvelteKit
npx sv create .
```

Nas opções interativas, selecione:

1. "SvelteKit demo" (para primeiro projeto)
2. "Yes, using TypeScript syntax"
3. Selecione as ferramentas recomendadas (prettier, eslint, playwright)
4. pnpm como gerenciador de pacotes

**IMPORTANTE**: após a criação do projeto, instale as dependências. Pode ser que o Sveltekit rode sem as todas as dependências instaladas. Na minha primeira tentativa eu ignorei um _warning_ e isso em custou uma hora de pesquisa para entender o que estava acontecendo.

```shell
pnpm install
```

## Instalação do Tailwind e Flowbite

Após confirmar que o projeto base está funcionando, vamos adicionar Tailwind e Flowbite:

```shell
# Instala o Tailwind CSS
npx svelte-add@latest tailwindcss

# Do you want to use typography plugin?
# Selecione Yes e clique enter

# Reinstala as dependências após adicionar o Tailwind
pnpm install

# Instala o Flowbite e seus componentes
pnpm add -D flowbite-svelte flowbite flowbite-svelte-icons
```

## Configuração dos Arquivos do Projeto

Crie ou atualize os seguintes arquivos de configuração.

O arquivo `vite.config.ts`:

```shell
import adapter from '@sveltejs/adapter-node';

/** @type {import('@sveltejs/kit').Config} */
const config = {
   kit: {
       adapter: adapter()
   }
};

export default config;
```

O arquivo `svelte.config.js`:

```shell
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
   plugins: [sveltekit()],
   server: {
       host: '0.0.0.0',
       port: 5173
   }
});
```

O arquivo `tailwind.config.ts`:

```shell
import flowbitePlugin from 'flowbite/plugin';
import type { Config } from 'tailwindcss';

export default {
   content: [
       './src/**/*.{html,js,svelte,ts}',
       './node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}'
   ],
   darkMode: 'selector',
   theme: {
       extend: {
           colors: {
               primary: {
                   50: '#FFF5F2',
                   100: '#FFF1EE',
                   200: '#FFE4DE',
                   300: '#FFD5CC',
                   400: '#FFBCAD',
                   500: '#FE795D',
                   600: '#EF562F',
                   700: '#EB4F27',
                   800: '#CC4522',
                   900: '#A5371B'
               }
           }
       }
   },
   plugins: [flowbitePlugin]
} satisfies Config;
```

## Configuração de Acesso Remoto

Configure o firewall para permitir acesso à porta de desenvolvimento:

```shell
# Libera a porta do servidor de desenvolvimento
sudo ufw allow 5173/tcp

# Verifica as regras do firewall
sudo ufw status
```

## Iniciando o Servidor de Desenvolvimento

Agora podemos iniciar o servidor com acesso remoto habilitado:

```shell
# Inicia o servidor de desenvolvimento
pnpm dev --host
```

Após seguir estes passos, você poderá acessar seu servidor de desenvolvimento em `http://SEU_IP_SERVIDOR:5173`. Lembre-se de que esta configuração é para desenvolvimento. Para produção, serão necessárias medidas adicionais de segurança e configuração adequada do servidor.

## Solução de Alguns dos Problemas encontrados

Se encontrar o erro "vite: not found" ou problemas similares:

```shell
# Verifica se estamos no diretório correto
pwd

# Verifica se package.json existe
ls package.json

# Se existe package.json mas não há node_modules:
pnpm install

# Se ainda houver problemas com o vite:
pnpm add -D vite @sveltejs/kit
pnpm install
```

Se encontrar problemas de permissão:

```shell
# Corrige as permissões do diretório do projeto
sudo chown -R $USER:$USER .

# Verifica se as permissões foram corrigidas
ls -la
```

Se o pnpm não estiver funcionando corretamente:

```shell
# Limpa o cache do npm
npm cache clean --force

# Reinstala o pnpm
sudo npm install -g pnpm

# Verifica a instalação
pnpm --version
```

## Comandos Úteis

Para parar o servidor: `Ctrl+C` se estiver no mesmo terminal em que o servidor foi inicializado ou:

```shell
# Localiza e encerra processo
pkill -f vite

# Ou ainda, usando a porta
kill $(lsof -t -i:5173)
```

Para remover a instalação: eu tive que fazer isso algumas vezes para entender os erros de compatibilidade entre as diversas versões dos módulos do Sveltekit, Flowbite e Tailwind.

```shell
# Remove o diretório do projeto
sudo rm -rf /var/www/lotofacil
sudo rm -rf /home/lotofacil

# Desinstala pnpm globalmente
sudo npm uninstall -g pnpm

# Remove Node.js e npm
sudo apt-get remove nodejs npm
sudo apt-get autoremove

# Remove repositório NodeSource
sudo rm /etc/apt/sources.list.d/nodesource.list*

# Limpa pacotes e configurações residuais
sudo apt-get clean
sudo apt-get autoclean

# Remove dependências não utilizadas
sudo apt-get autoremove

# Remove cache do npm se existir
rm -rf ~/.npm
```

## Na minha máquina funciona

