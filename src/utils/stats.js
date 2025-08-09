export function calcStats(games, player) {
  const stats = { games: games.length, wins: 0, losses: 0, draws: 0, gf: 0, ga: 0 };
  games.forEach(g => {
    const isP1 = g.Player1 === player;
    const my = isP1 ? g.GoalsPlayer1 : g.GoalsPlayer2;
    const opp = isP1 ? g.GoalsPlayer2 : g.GoalsPlayer1;
    stats.gf += my;
    stats.ga += opp;
    if (my > opp) stats.wins++;
    else if (my < opp) stats.losses++;
    else stats.draws++;
  });
  stats.avgF = stats.games ? (stats.gf / stats.games).toFixed(2) : 0;
  stats.avgA = stats.games ? (stats.ga / stats.games).toFixed(2) : 0;
  return stats;
}
