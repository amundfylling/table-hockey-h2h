import React from 'react';
import Plot from 'react-plotly.js';
import { useData } from '../context/DataContext';

export function ScoreLineChart() {
  const { games } = useData();
  const data = React.useMemo(() => {
    const sorted = [...games].sort((a, b) => new Date(a.Date) - new Date(b.Date));
    const x = sorted.map(g => g.Date);
    const p1 = sorted.map(g => g.GoalsPlayer1);
    const p2 = sorted.map(g => g.GoalsPlayer2);
    return [
      { x, y: p1, name: 'Player1', type: 'scatter', mode: 'lines+markers' },
      { x, y: p2, name: 'Player2', type: 'scatter', mode: 'lines+markers' }
    ];
  }, [games]);

  return (
    <Plot
      data={data}
      layout={{ title: 'Scores Over Time', xaxis: { title: 'Date' }, yaxis: { title: 'Goals' }, margin: { t: 40 } }}
      className="w-full"
    />
  );
}

export function WinLossBarChart() {
  const { games, player } = useData();
  const counts = React.useMemo(() => {
    let w = 0, l = 0, d = 0;
    games.forEach(g => {
      const isP1 = g.Player1 === player;
      const my = isP1 ? g.GoalsPlayer1 : g.GoalsPlayer2;
      const opp = isP1 ? g.GoalsPlayer2 : g.GoalsPlayer1;
      if (my > opp) w++;
      else if (my < opp) l++;
      else d++;
    });
    return { w, l, d };
  }, [games, player]);

  const data = [
    {
      x: ['Wins', 'Losses', 'Draws'],
      y: [counts.w, counts.l, counts.d],
      type: 'bar',
      marker: { color: ['#16a34a', '#dc2626', '#ca8a04'] }
    }
  ];

  return (
    <Plot
      data={data}
      layout={{ title: 'Results', xaxis: { title: '' }, yaxis: { title: 'Games' }, margin: { t: 40 } }}
      className="w-full"
    />
  );
}
