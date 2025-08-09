import React from 'react';
import { useData } from '../context/DataContext';
import { calcStats } from '../utils/stats';

export default function SummaryCards() {
  const { games, player } = useData();
  const stats = React.useMemo(() => calcStats(games, player), [games, player]);

  const Card = ({ label, value }) => (
    <div className="bg-white shadow rounded p-4 text-center">
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
    </div>
  );

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
      <Card label="Games" value={stats.games} />
      <Card label="Wins" value={stats.wins} />
      <Card label="Losses" value={stats.losses} />
      <Card label="Avg GF" value={stats.avgF} />
      <Card label="Avg GA" value={stats.avgA} />
    </div>
  );
}
