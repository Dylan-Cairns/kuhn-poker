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

Public history starts empty (`[]`) and legal actions are:

1. `[]`:
- `CHECK_OR_CALL` (interpreted as `check`)
- `BET_OR_RAISE` (interpreted as `bet`)

2. `["check"]`:
- `CHECK_OR_CALL` (interpreted as `check`)
- `BET_OR_RAISE` (interpreted as `bet`)

3. `["bet"]` or `["check", "bet"]`:
- `CHECK_OR_CALL` (interpreted as `call`)
- `FOLD` (interpreted as `fold`)

`FOLD` is illegal unless facing a bet.
`BET_OR_RAISE` is illegal when already facing a bet.

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
