# 2分動画ドラフト（研究概要）

## 提出ファイル名
- 最終動画ファイル名：`group12_research_overview.mp4`
- このドラフト名：`video_draft_group12.md`

## 動画の狙い（2分以内）
- テーマ：**階層的世界モデル Hieros の実態評価と限界の可視化**
- 伝える結論（3点）
  1. Visual Pinpadではハイパーパラメータ依存が強く，頑健性が低い
  2. Atariでは高スコアでも単純方策が多く，期待した階層性が確認しづらい
  3. 階層数増加で学習安定性が低下する

---

## そのまま使える動画構成（合計 1分50秒）

### 0:00–0:12（12秒）タイトル
**画面**
- タイトルスライド
- 「階層的世界モデルの現状と課題：Hierosの限界と将来展望」

**ナレーション**
- 「本研究では，階層的世界モデルHierosを対象に，性能と内部挙動を可視化して評価しました。」

---

### 0:12–0:30（18秒）背景・目的
**画面**
- 背景スライド（Dreamer / Director / Hieros を1枚で整理）
- 目的：性能向上の報告だけでなく，**本当に階層性が機能しているか**を検証

**ナレーション**
- 「世界モデルはサンプル効率に優れ，階層化で長期計画が期待されます。そこでHierosの性能だけでなく，サブゴールや行動の中身まで確認しました。」

---

### 0:30–0:58（28秒）Pinpad結果（定量 + 可視化）
**画面（推奨素材）**
- 学習曲線：`media/pinpad/subactor-update-sweep/sweep-episode-scores.png`
- 可視化GIF：`presentation-videos/subgoal-visualization/early-stage-96k.gif`
- 可視化GIF：`presentation-videos/subgoal-visualization/mid-stage-296k.gif`
- 可視化GIF：`presentation-videos/subgoal-visualization/late-stage-395k.gif`

**ナレーション**
- 「Visual Pinpadでは設定を変えると過程は変わるものの，最終性能は限定的でした。サブゴール可視化でも，長期シーケンスを一貫して提案する学習は確認しづらい結果でした。」

---

### 0:58–1:26（28秒）Atari結果（スコアと方策のギャップ）
**画面（推奨素材）**
- 学習曲線：`media/atari/atari_freeway-scores.png`
- 行動変化GIF：`presentation-videos/policy-behavior/early-policy-42k.gif`
- 行動変化GIF：`presentation-videos/policy-behavior/mid-policy-170k.gif`
- 行動変化GIF：`presentation-videos/policy-behavior/late-policy-368k.gif`

**ナレーション**
- 「Atariではスコア上は良好でも，行動フレームを確認すると単純動作の繰り返しが見られました。つまり，高スコアと階層的な意味ある方策学習が一致しない可能性があります。」

---

### 1:26–1:42（16秒）時間発展の要約
**画面（推奨素材）**
- `presentation-videos/timeline-progression/very-early-10k.gif`
- `presentation-videos/timeline-progression/comparison-232k.gif`

**ナレーション**
- 「時系列比較でも，性能向上と内部の階層的表現の成熟が必ずしも一致しないことが示唆されました。」

---

### 1:42–1:50（8秒）まとめ
**画面**
- 箇条書き3点（上の結論を再掲）

**ナレーション**
- 「以上より，Hierosには頑健性と解釈性の面で課題があり，動的抽象化と安定学習の改善が今後の重要課題です。」

---

## スライド最小構成（6枚）
1. タイトル
2. 背景・目的
3. Pinpad結果
4. Atari結果
5. 時系列比較
6. 結論・今後

## 収録時の注意
- 再生速度は標準（必要ならGIFを等速でループ）
- 字幕を入れる場合は1行20〜30文字程度
- 2分制約のため，各スライドは10〜30秒で切替

## 提出前チェック
- [ ] 2分以内（推奨 1:45〜1:55）
- [ ] シミュレーション/実機結果を動画内に含む（本ドラフトではGIFで満たす）
- [ ] ファイル名にグループ番号を含む（`group12_research_overview.mp4`）
