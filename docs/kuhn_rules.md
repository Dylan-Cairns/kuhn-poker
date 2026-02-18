# Kuhn Poker Rules (Roadmap Item 1)

This project implements a 2-player variant of standard Kuhn Poker with one hand per episode.

## Deck and Cards

- Deck has 3 cards: `J`, `Q`, `K`
- Card strength: `K > Q > J`
- At hand start, each player is dealt exactly 1 private card
- The remaining card is not observed

## Antes and Pot

- Both players ante `1` chip at hand start
- Initial pot is `2`
- Single betting round only
- At most one bet is allowed (no raises)

## Turn Order and Public Action History

`player_0` always acts first.

## AEC Phase State Machine

Environment phases are explicit and finite:

- `deal` -> internal setup, no agent action
- `p0_act` -> `player_0` first decision
- `p1_act` -> `player_1` decision after `player_0` checked
- `p0_response` -> `player_0` response to `player_1` bet
- `p1_response` -> `player_1` response to `player_0` bet
- `terminal` -> hand is over, rewards assigned

Phase transitions:

- `deal -> p0_act`
- `p0_act --check--> p1_act`
- `p0_act --bet--> p1_response`
- `p1_act --check--> terminal` (showdown)
- `p1_act --bet--> p0_response`
- `p0_response|p1_response --call--> terminal` (showdown)
- `p0_response|p1_response --fold--> terminal` (bettor wins)

## Action Encoding and Masks (Roadmap Item 3)

Fixed discrete action IDs:

- `0 = CHECK_OR_CALL`
- `1 = BET`
- `2 = FOLD`

Mask semantics:

- Mask shape: `(3,)`
- Mask dtype: `int8`
- `1` means legal, `0` means illegal

By phase:

- `p0_act` / `p1_act` -> `[1, 1, 0]` (check/bet)
- `p0_response` / `p1_response` -> `[1, 0, 1]` (call/fold)
- `terminal` or not-current agent -> `[0, 0, 0]`

## Observation Encoding (Roadmap Item 3)

Observation is a fixed binary vector of length `10`:

- Indices `[0:3]`: private card one-hot (`J, Q, K`)
- Indices `[3:8]`: public history one-hot
- Indices `[8:10]`: current actor one-hot (`player_0`, `player_1`)

History one-hot mapping:

- index `3` (`history_id=0`): `[]`
- index `4` (`history_id=1`): `["check"]`
- index `5` (`history_id=2`): `["bet"]`
- index `6` (`history_id=3`): `["check", "bet"]`
- index `7` (`history_id=4`): terminal/other complete history

At terminal, actor bits are `[0, 0]`.

Public history starts empty (`[]`) and legal actions are:

1. `[]`:
- `CHECK_OR_CALL` (interpreted as `check`)
- `BET` (interpreted as `bet`)

2. `["check"]`:
- `CHECK_OR_CALL` (interpreted as `check`)
- `BET` (interpreted as `bet`)

3. `["bet"]` or `["check", "bet"]`:
- `CHECK_OR_CALL` (interpreted as `call`)
- `FOLD` (interpreted as `fold`)

`FOLD` is illegal unless facing a bet.
`BET` is illegal when already facing a bet.

## Terminal Conditions

The hand ends on one of these public histories:

- `["check", "check"]` -> showdown
- `["bet", "fold"]` -> bettor wins
- `["bet", "call"]` -> showdown
- `["check", "bet", "fold"]` -> bettor wins
- `["check", "bet", "call"]` -> showdown

## Payoff Convention

Rewards are net chip change (zero-sum per hand):

- Winner reward: `pot - winner_contribution`
- Loser reward: `-loser_contribution`

Equivalent outcomes:

- Check-check showdown: winner `+1`, loser `-1`
- Bet-fold: bettor `+1`, folder `-1`
- Bet-call showdown: winner `+2`, loser `-2`

## Information Model

At decision time, each player can use:

- Their private card
- Public betting history

Players cannot observe the opponent's private card before terminal showdown resolution.
