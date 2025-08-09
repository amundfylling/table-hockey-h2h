import { calcStats } from '../src/utils/stats';

describe('calcStats', () => {
  test('computes wins, losses, draws and averages', () => {
    const games = [
      { Player1: 'A', Player2: 'B', GoalsPlayer1: 3, GoalsPlayer2: 1 },
      { Player1: 'B', Player2: 'A', GoalsPlayer1: 2, GoalsPlayer2: 2 },
      { Player1: 'A', Player2: 'C', GoalsPlayer1: 0, GoalsPlayer2: 1 }
    ];
    const stats = calcStats(games, 'A');
    expect(stats.games).toBe(3);
    expect(stats.wins).toBe(1);
    expect(stats.losses).toBe(1);
    expect(stats.draws).toBe(1);
    expect(stats.avgF).toBe('1.67');
    expect(stats.avgA).toBe('1.33');
  });
});
