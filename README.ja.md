# yomu

[English](README.md)

AIエージェント向けフロントエンドコード検索。名前がわからなくても、概念でコードを見つけられる。

## 課題

AIエージェントが [vercel/ai](https://github.com/vercel/ai)（3,535ファイル）のチャットストリーミングフックを探す必要がある。しかし関数名が `useChat` であることを知らない。

**従来のエージェントワークフロー:**

```
glob "**/chat*"       → 12ファイル。どれも違う（実際は use-chat.ts）
grep "stream.*hook"   → 0ファイル
grep "chat"           → 結果が多すぎる。数件読んでみる…
read packages/react/src/use-chat.ts  → 見つけた。何をインポートしてる？
read packages/ai/src/ui/process-ui-message-stream.ts → これでコンテキストが揃った
```

3〜5回のツール呼び出し、試行錯誤、コンテキストウィンドウにノイズが溜まる。

**yomu なら:**

```
explorer("streaming chat hooks")

  #1  useChat [hook] — packages/react/src/use-chat.ts:58
      関数本体、インポート、隣接する型定義をすべて含む。

  #2  useStreamableValue [hook] — packages/rsc/src/streamable-value/...
  #3  useSharedChatContext [hook] — examples/ai-e2e-next/.../chat-context.tsx
```

1回の呼び出し。実装が最初の結果として返る — "useChat" を含む130ファイル、合計9,015チャンクのインデックスの中から。

各結果にはコード本体、ファイルのインポート、隣接する定義が含まれる。追加の read は不要。

## yomu を使うべき場面（と使わない場面）

**yomu を使う:**

- コードの名前がわからない — 「エラーバウンダリのロジック」「データフェッチ層」
- grep の結果が多すぎて、125個のテストファイルではなく実装が欲しい
- コード + インポート + 関連型を1レスポンスで取得したい

**grep/glob を使う:**

- 正確な名前を知っている — `grep "useAuth"` の方が速くて正確
- 正規表現やリテラル文字列の検索が必要
- コードベースが小さくて馴染みがある

yomu は grep の代替ではない。grep が対応できないケース、つまり概念による検索をカバーする。

## セットアップ

### インストール

```sh
brew install thkt/tap/yomu
```

またはソースからビルド（Rust 1.85+ が必要）:

```sh
cargo build --release
```

### 設定

[Gemini API キー](https://aistudio.google.com/apikey)が必要（無料枠で動作）。

MCP クライアントの設定に追加（例: Claude Code の `~/.claude/.mcp.json`）:

```json
{
  "mcpServers": {
    "yomu": {
      "command": "yomu",
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

手動インデックスは不要。`explorer` が初回呼び出し時に自動インデックスする。

## ツール

### `explorer` — 概念で検索

ランク付けされた結果をフルコンテキスト付きで返す。各結果に含まれるもの:

| 内容              | 理由                                             |
| ----------------- | ------------------------------------------------ |
| コード本体        | 追加の `read` が不要                             |
| ファイルの import | 別ファイルを開かずに依存関係のコンテキストを把握 |
| 隣接する定義      | 同一ファイル内の他の関数・型                     |
| チャンクタイプ    | component / hook / type_def / css_rule           |

### `impact` — 変更の影響範囲

対象のファイルまたはシンボルに依存しているファイルを表示。

```
> impact("packages/ai/src/ui/ui-messages.ts:UIMessage", depth=2)

  Direct symbol references: 34 files
  Depth 1: 37 files
  Depth 2: 18 files
  Total: 55 dependent files
```

vercel/ai での実際の出力。1回の呼び出しで手動のインポート追跡を置き換える。

### `index` / `rebuild` / `status`

- `index` — チャンクインデックスを更新。API呼び出しなし、3,535ファイルで約2.5秒。通常は不要 — `explorer` が自動インデックスする。
- `rebuild` — ゼロから完全に再パース。
- `status` — ファイル数、チャンク数、エンベディングカバレッジ、リファレンス数を表示。

## 仕組み

```
ソースファイル → tree-sitter AST → セマンティックチャンク → Gemini エンベディング → ハイブリッド検索
```

**インデックス作成** — tree-sitter が関数・コンポーネント・型の境界でコードを分割。各チャンクが1つの検索単位になる。インポートグラフも同じパスで構築。vercel/ai の場合: 3,535ファイル → 9,015チャンク + 5,026インポートリファレンスを2.5秒で、API呼び出しゼロ。

**エンベディング** — チャンクは Gemini（無料枠）経由でインクリメンタルにエンベディングされる。`explorer` 呼び出しごとに50チャンク、インポート数の多い順に優先 — 最も使われているコードが先に検索可能になる。事前ビルド不要。

**検索** — 3層ハイブリッド: ベクトル類似度 → 名前・パスマッチング → FTS5 全文検索。結果は IDF 加重キーワードスコアリングで再ランク付け。頻繁にインポートされるファイルは上位に、テストファイルは下位に配置。

## 対応ファイルタイプ

| タイプ           | パーサー    | チャンクタイプ                              |
| ---------------- | ----------- | ------------------------------------------- |
| TypeScript / TSX | tree-sitter | component, hook, type_def, test_case, other |
| JavaScript / JSX | tree-sitter | component, hook, type_def, test_case, other |
| CSS              | tree-sitter | css_rule (selectors, @media, @keyframes)    |
| HTML             | tree-sitter | html_element                                |

その他のファイルはオーバーラップ付きの文字数ベースチャンキングにフォールバック。

## 制限事項

- **Gemini 無料枠のレート制限** — 100 RPM、1日1,500リクエスト。プロジェクトをまたいだ大量使用で日次クォータを使い切る可能性あり（太平洋時間の深夜にリセット）。
- **SCSS/Sass 非対応** — プレーン CSS のみ。
- **コールドスタート** — 初回の `explorer` 呼び出しはチャンキング + 初期エンベディングに数秒かかる。
- **大きなファイルはスキップ** — 1 MB を超えるファイルはインデックス対象外。

## アーキテクチャ

```
src/
├── main.rs              MCP サーバー（stdio トランスポート）
├── lib.rs               クレートルート、パブリック API
├── config.rs            ランタイム設定
├── tools/               MCP ツールハンドラー（explorer, index, rebuild, impact, status）
├── indexer/
│   ├── mod.rs           オーケストレーション: インクリメンタルインデックス、エンベッド予算
│   ├── chunker/         tree-sitter AST → セマンティックチャンク
│   ├── embedder.rs      Gemini エンベディングクライアント（batchEmbedContents）
│   └── walker.rs        ファイル探索、.gitignore フィルタリング
├── resolver.rs          インポートパス解決（tsconfig エイリアス、index.ts プロービング）
├── query.rs             ハイブリッド検索 + IDF 再ランキング
└── storage/
    ├── mod.rs           スキーマ、CRUD、型定義
    ├── search.rs        ベクトル類似度、名前マッチング、FTS5
    └── graph.rs         インポートグラフ走査、依存先、隣接定義
```

シングルバイナリ、ランタイム依存ゼロ。SQLite と sqlite-vec は静的リンク。

## ライセンス

Apache-2.0
