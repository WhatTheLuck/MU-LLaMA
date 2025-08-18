# cd MU-LLaMA
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/7B.pth?download=true
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/.gitattributes?download=true
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/README.md?download=true
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/checkpoint.pth?download=true

# cd MU-LLaMA/LLaMA
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/llama.sh?download=true
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/tokenizer.model?download=true
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/tokenizer_checklist.chk?download=true

# cd MU-LLaMA/LLaMA/7B
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/checklist.chk?download=
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/consolidated.00.pth?download=true
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/params.json?download=true
# https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/test.txt?download=true

#!/bin/bash

# MU-LLaMA 模型文件批量下载脚本

echo "🚀 开始批量下载 MU-LLaMA 模型文件"

# 函数：安全下载文件
download_file() {
    local url=$1
    local filepath=$2
    
    # 如果文件已存在，跳过下载
    if [[ -f "$filepath" ]]; then
        echo "⏭️  文件已存在，跳过: $filepath"
        return 0
    fi
    
    echo "⬇️  下载: $filepath"
    
    # 使用 wget 或 curl 下载文件
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$filepath" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "$filepath" "$url"
    else
        echo "❌ 错误: 需要安装 wget 或 curl"
        return 1
    fi
    
    if [[ $? -eq 0 ]]; then
        echo "✅ 下载完成: $filepath"
        return 0
    else
        echo "❌ 下载失败: $filepath"
        return 1
    fi
}

# 下载 MU-LLaMA 根目录文件
echo ""
echo "📂 处理目录: MU-LLaMA"
mkdir -p MU-LLaMA

download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/7B.pth?download=true" "MU-LLaMA/7B.pth"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/.gitattributes?download=true" "MU-LLaMA/.gitattributes"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/README.md?download=true" "MU-LLaMA/README.md"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/checkpoint.pth?download=true" "MU-LLaMA/checkpoint.pth"

# 下载 MU-LLaMA/LLaMA 目录文件
echo ""
echo "📂 处理目录: MU-LLaMA/LLaMA"
mkdir -p MU-LLaMA/LLaMA

download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/llama.sh?download=true" "MU-LLaMA/LLaMA/llama.sh"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/tokenizer.model?download=true" "MU-LLaMA/LLaMA/tokenizer.model"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/tokenizer_checklist.chk?download=true" "MU-LLaMA/LLaMA/tokenizer_checklist.chk"

# 下载 MU-LLaMA/LLaMA/7B 目录文件
echo ""
echo "📂 处理目录: MU-LLaMA/LLaMA/7B"
mkdir -p MU-LLaMA/LLaMA/7B

download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/checklist.chk?download=true" "MU-LLaMA/LLaMA/7B/checklist.chk"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/consolidated.00.pth?download=true" "MU-LLaMA/LLaMA/7B/consolidated.00.pth"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/params.json?download=true" "MU-LLaMA/LLaMA/7B/params.json"
download_file "https://huggingface.co/mu-llama/MU-LLaMA/resolve/main/LLaMA/7B/test.txt?download=true" "MU-LLaMA/LLaMA/7B/test.txt"

echo ""
echo "🎉 下载脚本执行完成！"
echo ""
echo "文件结构："
tree MU-LLaMA 2>/dev/null || find MU-LLaMA -type f | sort