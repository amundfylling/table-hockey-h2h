import React from 'react';
import { useData } from '../context/DataContext';

export default function Filters() {
  const { player, setPlayer, season, setSeason, players, seasons } = useData();

  return (
    <div className="flex flex-wrap gap-4 mb-4">
      <select value={player} onChange={e => setPlayer(e.target.value)} className="border p-2 rounded">
        <option value="">All Players</option>
        {players.map(p => (
          <option key={p} value={p}>{p}</option>
        ))}
      </select>
      <select value={season} onChange={e => setSeason(e.target.value)} className="border p-2 rounded">
        <option value="">All Seasons</option>
        {seasons.map(s => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>
    </div>
  );
}
