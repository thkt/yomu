[English](README.md) | **日本語**

# yomu

AIエージェント向けフロントエンドコード検索。名前がわからなくても、概念でコードを見つけられます。

## 課題

関数名が `useChat` だとわからない状態で、[vercel/ai](https://github.com/vercel/ai)（3,535ファイル）からチャットストリーミングフックを探したいとき、どうすればよいでしょうか。

従来のエージェントワークフローでは、以下のようになります。

```text
glob "**/chat*"                                      → 12ファイル。どれも違う（実際は use-chat.ts）
grep "stream.*hook"                                  → 0ファイル
grep "chat"                                          → 結果が多すぎる。数件読んでみる…
read packages/react/src/use-chat.ts                  → 見つけた。何をインポートしてる？
read packages/ai/src/ui/process-ui-message-stream.ts → これでコンテキストが揃った
```

3〜5回のツール呼び出し、試行錯誤、コンテキストウィンドウにノイズが溜まります。

yomuならこれが1回で済みます。

```markdown
yomu search "streaming chat hooks"

## packages/react/src/use-chat.ts

Imports: @ai-sdk/provider-utils, @ai-sdk/ui-utils
Siblings: UseChatOptions [type_def], UseChatHelpers [type_def]

1. useChat [hook] — 58:210 (similarity: 0.85)
   export function useChat({ api, ...options }: UseChatOptions): UseChatHelpers {
   ...関数本体...
   }

## packages/rsc/src/streamable-value/use-streamable-value.ts

2. useStreamableValue [hook] — 12:45 (similarity: 0.72)
   ...

## examples/ai-e2e-next/.../chat-context.tsx

3. useSharedChatContext [hook] — 8:22 (similarity: 0.68)
   ...
```

`useChat`を含む130ファイル、9,015チャンクのインデックスから、1回の呼び出しで実装が最初の結果として返ります。

各結果にはコード本体、ファイルのインポート、隣接する定義が含まれます。追加のreadは不要です。

## なぜ grep ではないのか

[Claude Codeの開発者](https://zenn.dev/acntechjp/articles/c1296f425baf03)によると、Agentic search（モデルにglobとgrepを反復的に使わせる手法）はRAGを上回ったそうです。実際、エージェントが別のキーワードでリトライしたりディレクトリ構造から当たりをつけたりできるなら、grepは非常にうまく機能します。

yomuはそのワークフローと競合しない、反復回数を減らすためのツールです。

| アプローチ        | 呼び出し回数 | コンテキストウィンドウへの影響            |
| ----------------- | ------------ | ----------------------------------------- |
| grep/glob（反復） | 3〜5回       | ミスのたびにノイズが増える                |
| yomu search       | 1回          | コード + import + 隣接定義を1レスポンスで |

インデックスの同期遅延、陳腐化したエンベディング、コールドスタートなどRAGの定番の問題には対処済みです。

| 対策                           | 内容                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------ |
| 同期遅延なし                   | `search` のたびにインデックスの鮮度をチェックし、ファイルが変更されていれば自動で再チャンクする  |
| APIキー不要                    | ローカルエンベディングモデル、モデル未インストールまたは利用不可時はFTS5全文検索にフォールバック |
| インクリメンタルエンベディング | search呼び出しごとに50チャンク、インポート数の多いファイルから優先。事前ビルド不要               |

名前がわかっているならgrepを使うのが良いでしょう。yomuはその手前、概念でわかっても識別子を知らない瞬間のためのツールです。

## yomuとgrepの比較

| yomu                                                  | grep/glob                                             |
| ----------------------------------------------------- | ----------------------------------------------------- |
| コードの名前がわからない                              | 正確な名前を知っている（`grep "useAuth"` の方が速い） |
| grepの結果が多すぎて、テストではなく実装が欲しい      | 正規表現やリテラル文字列の検索が必要                  |
| コード + インポート + 関連型を1レスポンスで取得したい | コードベースが小さくて馴染みがある                    |

## セットアップ

### インストール

```sh
brew install thkt/tap/yomu
```

またはソースからビルドできます（Rust 1.85+が必要）。

```sh
cargo build --release
```

### 設定

セマンティック検索にはローカルエンベディングモデル（[Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-310m)、約1.2 GB）を使用します。モデルがローカルキャッシュに存在する場合、`search` は自動的に利用します。APIキーは不要です。

モデルが未インストールまたは利用不可の場合、`search` はテキスト検索モード（FTS5）にフォールバックします。その他のコマンド（`index`、`rebuild`、`impact`、`status`）はモデル不要です。

手動インデックスは不要です。`search` が初回呼び出し時に自動インデックスします。

#### プラットフォーム別ビルド

| プラットフォーム      | ビルドコマンド                                                  |
| --------------------- | --------------------------------------------------------------- |
| macOS (Apple Silicon) | `cargo build --release`（デフォルト: mlx バックエンド）         |
| Linux / x86           | `cargo build --release --no-default-features --features candle` |

## コマンド

### グローバルフラグ

| フラグ   | 説明                              |
| -------- | --------------------------------- |
| `--json` | JSON 形式で出力（全コマンド共通） |

`--json` はサブコマンドの前後どちらにも配置できます:

```sh
$ yomu --json status
{"files":42,"chunks":187,"embedded_chunks":187,"embeddable_chunks":187,"embed_percentage":100,"references":156,"last_indexed":"2025-03-29 01:23:45"}
```

### `yomu search [query]` — 概念で検索

ランク付けされた結果をフルコンテキスト付きで返します。各結果には以下が含まれます。

| 内容              | 理由                                             |
| ----------------- | ------------------------------------------------ |
| コード本体        | 追加の `read` が不要                             |
| ファイルの import | 別ファイルを開かずに依存関係のコンテキストを把握 |
| 隣接する定義      | 同一ファイル内の他の関数・型                     |
| チャンクタイプ    | component / hook / type_def / css_rule / rust_fn |

オプション:

| フラグ       | デフォルト | 説明                                                                          |
| ------------ | ---------- | ----------------------------------------------------------------------------- |
| `--limit`    | 10         | 最大件数（上限: 100）                                                         |
| `--offset`   | 0          | ページネーションオフセット（上限: 500）                                       |
| `--from`     | —          | ファイルまたはシンボルに類似するコードを検索（`src/foo.rs` または `src/foo.rs:my_fn`）。query は省略可 |
| `--no-embed` | false      | エンベディングをスキップして FTS5 のみで検索。`YOMU_EMBED=0` と同等           |

`--from` は保存済み埋め込みをそのまま検索クエリとして使うため、再埋め込みは不要です:

```sh
yomu search --from src/query/mod.rs           # このファイルに類似するファイルを検索
yomu search --from src/query/mod.rs:rerank    # この関数に類似するシンボルを検索
yomu search --from src/query/mod.rs "filter"  # ハイブリッド: ファイル類似 + FTS "filter"
```

### `yomu impact <target>` — 変更の影響範囲

対象のファイルまたはシンボルに依存しているファイルを表示します。

```sh
$ yomu impact "packages/ai/src/ui/ui-messages.ts" --symbol UIMessage --depth 2

## Impact analysis: `packages/ai/src/ui/ui-messages.ts`

### Direct symbol references
- packages/ai/src/ui/process-ui-message-stream.ts
- packages/react/src/use-chat.ts
  ...34 files

### All transitive dependents
#### Depth 1
- packages/ai/src/ui/process-ui-message-stream.ts
  ...37 files
#### Depth 2
- packages/react/src/use-chat.ts
  ...18 files

Total: 55 dependent file(s)
```

vercel/aiでの実際の出力です。1回の呼び出しで手動のインポート追跡を置き換えられます。

オプション: `--symbol`（任意、特定のエクスポートでフィルター）、`--depth`（デフォルト: 3、最大: 10）

`--json` は構造化レスポンスを返します。`dependents[].references` は直接依存（depth=1）のエッジに対して埋まり、source ファイルからの `(ref_kind, via_symbol)` の組み合わせを列挙します。推移依存（depth>=2）は中間ファイルを経由するため `references` は `[]` です:

```json
{
  "target": "src/storage.rs",
  "in_index": true,
  "dependents": [
    {
      "file_path": "src/indexer.rs",
      "depth": 1,
      "references": [
        {"ref_kind": "named", "via_symbol": "Db"},
        {"ref_kind": "named", "via_symbol": "open_db"}
      ]
    },
    {
      "file_path": "src/main.rs",
      "depth": 2,
      "references": []
    }
  ],
  "symbol_refs": [],
  "total": 2
}
```

`ref_kind` は `named` / `default` / `namespace` / `type_only` / `side_effect` のいずれか。`via_symbol` は import 元で個別シンボルが名前付けされない namespace / side-effect import では `null` になります。

### `yomu index` / `yomu rebuild` / `yomu status`

| コマンド  | 内容                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| `index`   | チャンクインデックスを更新する。API呼び出しなし、3,535ファイルで約2.5秒。通常は不要 — `search` が自動実行する |
| `rebuild` | ゼロからすべて再パースする                                                                                    |
| `status`  | ファイル数、チャンク数、エンベディングカバレッジ、リファレンス数を表示する                                    |

## 仕組み

```text
ソースファイル → tree-sitter AST → セマンティックチャンク → ローカルエンベディング (Ruri v3) → ハイブリッド検索
```

| ステージ         | 内容                                                                                                                                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| インデックス作成 | tree-sitterが関数・コンポーネント・型の境界でコードを分割する。各チャンクが1つの検索単位になる。インポートグラフも同じパスで構築。vercel/aiの場合: 3,535ファイル → 9,015チャンク + 5,026インポートリファレンスを2.5秒で、API呼び出しゼロ |
| エンベディング   | チャンクはローカルモデル（Ruri v3、310Mパラメータ）でインクリメンタルにエンベディングされる。`search` 呼び出しごとに50チャンク、インポート数の多い順で優先。もっとも使われているコードから検索可能になる。事前ビルド不要                 |
| 検索             | 3層ハイブリッド: ベクトル類似度 → 名前・パスマッチング → FTS5全文検索。IDF加重キーワードスコアリングで再ランク付け。頻繁にインポートされるファイルは上位に、テストファイルは下位に配置                                                   |

## 対応ファイルタイプ

| タイプ           | パーサー    | チャンクタイプ                                         |
| ---------------- | ----------- | ------------------------------------------------------ |
| TypeScript / TSX | tree-sitter | component, hook, type_def, test_case, other            |
| JavaScript / JSX | tree-sitter | component, hook, type_def, test_case, other            |
| Rust             | tree-sitter | rust_fn, rust_struct, rust_enum, rust_trait, rust_impl |
| CSS              | tree-sitter | css_rule (selectors, @media, @keyframes)               |
| HTML             | tree-sitter | html_element                                           |

その他のファイルはオーバーラップ付きの文字数ベースチャンキングにフォールバックします。

## 制限事項

| 制限                     | 内容                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| モデル自動DLなし         | 約1.2 GBのエンベディングモデルは事前にキャッシュが必要。`search` は自動ダウンロードしない |
| SCSS/Sass非対応          | プレーンCSSのみ                                                                           |
| コールドスタート         | 初回の `search` 呼び出しはチャンキング + 初期エンベディングに数秒かかる                   |
| 大きなファイルはスキップ | 1 MBを超えるファイルはインデックス対象外                                                  |
| エンベディング無効化     | `yomu search --no-embed` を指定するか `YOMU_EMBED=0` を設定。`search` はテキスト検索のみで動作する |

## アーキテクチャ

```text
src/
├── main.rs              CLI エントリーポイント（clap）
├── lib.rs               クレートルート、パブリック API
├── config.rs            ランタイム設定
├── modernbert.rs        ModernBERT モデル（mlx バックエンド）
├── tools/               アプリケーションファサード — コマンドごとに indexer, query, storage を統合
├── indexer/
│   ├── mod.rs           オーケストレーション: インクリメンタルインデックス、エンベッド予算
│   ├── chunker/         tree-sitter AST → セマンティックチャンク
│   ├── embedder.rs      ローカルエンベディング（ModernBERT/Ruri v3、mlx-rs or candle）
│   └── walker.rs        ファイル探索、.gitignore フィルタリング
├── resolver.rs          インポートパス解決（tsconfig エイリアス、index.ts プロービング）
├── query/               ハイブリッド検索 + IDF 再ランキング
└── storage/
    ├── mod.rs           スキーマ、CRUD、型定義
    ├── search.rs        ベクトル類似度、名前マッチング、FTS5
    ├── embed.rs         エンベディングストレージ、vec_chunks
    └── graph.rs         インポートグラフ走査、依存先、隣接定義
```

シングルバイナリ、ランタイム依存ゼロ。SQLiteとsqlite-vecは静的リンクです。

## ライセンス

MIT
