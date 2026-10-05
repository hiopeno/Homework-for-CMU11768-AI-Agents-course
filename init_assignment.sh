#!/usr/bin/env bash
set -e

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

TARGET_DIR="$1"
if [ -z "$TARGET_DIR" ]; then
    read -rp "请输入作业目录名称 (例如 assignment-2): " TARGET_DIR
fi
TARGET_DIR="${TARGET_DIR%/}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ 错误: 目录 '$TARGET_DIR' 不存在！"
    exit 1
fi

echo "=========================================="
echo "🚀 开始全自动初始化: $TARGET_DIR"
echo "=========================================="

# 1. 消除 Git 套娃：删除解压自带的 .git
if [ -d "$TARGET_DIR/.git" ]; then
    echo "🧹 移除内层 '$TARGET_DIR/.git'，避免嵌套冲突..."
    rm -rf "$TARGET_DIR/.git"
fi

# 2. 配置根目录 .gitignore
ensure_gitignore() {
    local entry="$1"
    if ! grep -Fxq "$entry" .gitignore 2>/dev/null; then
        echo "$entry" >> .gitignore
    fi
}
touch .gitignore
ensure_gitignore "**/.venv/"
ensure_gitignore "**/__pycache__/"

# 3. 自动克隆 .gitmodules 中的沙盒仓库（默认拉取最新）
GITMODULES_FILE="$TARGET_DIR/.gitmodules"
SUB_DIRS=()

if [ -f "$GITMODULES_FILE" ]; then
    SUBMODULE_PATHS=$(git config -f "$GITMODULES_FILE" --get-regexp '\.path$' | awk '{print $2}')
    for SUB_PATH in $SUBMODULE_PATHS; do
        SUB_NAME=$(git config -f "$GITMODULES_FILE" --name-only --get-regexp "\.path$" | grep -F ".$SUB_PATH" | sed 's/^submodule\.//;s/\.path$//')
        CLONE_URL=$(git config -f "$GITMODULES_FILE" "submodule.${SUB_NAME}.url")
        FULL_SUB_DIR="$TARGET_DIR/$SUB_PATH"
        SUB_DIRS+=("$FULL_SUB_DIR")

        echo "📦 发现评测沙盒: $SUB_PATH ($CLONE_URL)"
        rm -rf "$FULL_SUB_DIR"
        git clone --quiet "$CLONE_URL" "$FULL_SUB_DIR"

        # 忽略沙盒目录，防止大仓库把它当成子模块
        ensure_gitignore "$FULL_SUB_DIR/"
        git rm -r --cached "$FULL_SUB_DIR" 2>/dev/null || true
    done
fi

# 4. 注释 Makefile 中的 git submodule update
MAKEFILE="$TARGET_DIR/Makefile"
if [ -f "$MAKEFILE" ] && grep -q "git submodule update" "$MAKEFILE"; then
    sed -i.bak 's/^\([[:space:]]*\)git submodule update/# \1git submodule update/' "$MAKEFILE"
    rm -f "${MAKEFILE}.bak"
fi

# 5. 自愈校验：运行 make setup，若版本不匹配则自动提取 commit 并切换
echo "▶️ 正在运行 make setup 并自动校准版本..."
cd "$TARGET_DIR"

set +e
SETUP_LOG=$(make setup 2>&1)
SETUP_EXIT=$?
set -e

if [ $SETUP_EXIT -ne 0 ]; then
    # 提取类似 "defined against 4aaabca1194e" 的哈希值
    TARGET_COMMIT=$(echo "$SETUP_LOG" | grep -oE "defined against [0-9a-fA-F]+" | awk '{print $3}' | head -n 1)

    if [ -n "$TARGET_COMMIT" ]; then
        echo "💡 检测到任务要求的基准版本: $TARGET_COMMIT"
        # 切换所有沙盒仓库到该 commit
        for S_DIR in "${SUB_DIRS[@]}"; do
            (cd "$REPO_ROOT/$S_DIR" && git checkout --quiet "$TARGET_COMMIT" 2>/dev/null || true)
        done
        echo "🔄 已自动对齐版本，正在重新执行 make setup..."
        make setup
    else
        echo "❌ make setup 遇到非版本匹配类错误，输出如下："
        echo "$SETUP_LOG"
        exit 1
    fi
else
    echo "$SETUP_LOG"
fi

echo "=========================================="
echo "🎉 初始化完成！环境与沙盒版本已自动对齐。"
echo "=========================================="