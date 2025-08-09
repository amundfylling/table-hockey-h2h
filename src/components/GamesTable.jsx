import React from 'react';
import { useData } from '../context/DataContext';

export default function GamesTable() {
  const { games } = useData();
  const [search, setSearch] = React.useState('');
  const [sortKey, setSortKey] = React.useState('');
  const [sortAsc, setSortAsc] = React.useState(true);
  const [page, setPage] = React.useState(0);
  const pageSize = 20;

  const headers = games.length ? Object.keys(games[0]) : [];

  const filtered = React.useMemo(() => {
    return games.filter(g =>
      Object.values(g).some(v => String(v).toLowerCase().includes(search.toLowerCase()))
    );
  }, [games, search]);

  const sorted = React.useMemo(() => {
    const arr = [...filtered];
    if (sortKey) {
      arr.sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        if (av === bv) return 0;
        return av > bv ? (sortAsc ? 1 : -1) : (sortAsc ? -1 : 1);
      });
    }
    return arr;
  }, [filtered, sortKey, sortAsc]);

  const pageCount = Math.ceil(sorted.length / pageSize) || 1;
  const pageData = sorted.slice(page * pageSize, (page + 1) * pageSize);

  function handleSort(h) {
    if (sortKey === h) setSortAsc(!sortAsc);
    else { setSortKey(h); setSortAsc(true); }
  }

  return (
    <div className="bg-white p-4 shadow rounded">
      <input
        type="text"
        placeholder="Search"
        value={search}
        onChange={e => { setSearch(e.target.value); setPage(0); }}
        className="mb-2 p-2 border rounded w-full"
      />
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              {headers.map(h => (
                <th
                  key={h}
                  onClick={() => handleSort(h)}
                  className="border-b text-left p-2 cursor-pointer select-none"
                >
                  {h}{sortKey === h ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageData.map((row, i) => (
              <tr key={i} className="odd:bg-gray-50">
                {headers.map(h => (
                  <td key={h} className="p-2 border-b whitespace-nowrap">{String(row[h])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex justify-between mt-2">
        <button
          disabled={page === 0}
          onClick={() => setPage(p => Math.max(p - 1, 0))}
          className="px-2 py-1 border rounded disabled:opacity-50"
        >
          Prev
        </button>
        <span>Page {page + 1} / {pageCount}</span>
        <button
          disabled={page + 1 >= pageCount}
          onClick={() => setPage(p => Math.min(p + 1, pageCount - 1))}
          className="px-2 py-1 border rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}
