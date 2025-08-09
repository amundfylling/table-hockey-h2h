import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

const DataContext = createContext();

export function DataProvider({ children }) {
  const [games, setGames] = useState([]);
  const [player, setPlayer] = useState('');
  const [season, setSeason] = useState('');

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/api/games');
        if (!res.ok) throw new Error('api unavailable');
        setGames(await res.json());
      } catch (err) {
        fetch('/data/combined_matches.json')
          .then(r => r.json())
          .then(setGames)
          .catch(e => console.error('Failed to load data', e));
      }
    }
    load();
  }, []);

  const players = useMemo(() => {
    const set = new Set();
    games.forEach(g => {
      set.add(g.Player1);
      set.add(g.Player2);
    });
    return Array.from(set).sort();
  }, [games]);

  const seasons = useMemo(() => {
    const set = new Set(games.map(g => new Date(g.Date).getFullYear()));
    return Array.from(set).sort();
  }, [games]);

  const filteredGames = useMemo(() => {
    return games.filter(g => {
      const year = new Date(g.Date).getFullYear();
      const matchPlayer = player ? g.Player1 === player || g.Player2 === player : true;
      const matchSeason = season ? year === Number(season) : true;
      return matchPlayer && matchSeason;
    });
  }, [games, player, season]);

  return (
    <DataContext.Provider value={{ games: filteredGames, allGames: games, player, setPlayer, season, setSeason, players, seasons }}>
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  return useContext(DataContext);
}
