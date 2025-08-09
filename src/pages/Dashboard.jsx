import React from 'react';
import Filters from '../components/Filters';
import SummaryCards from '../components/SummaryCards';
import { ScoreLineChart, WinLossBarChart } from '../components/Charts';
import GamesTable from '../components/GamesTable';

export default function Dashboard() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Table Hockey H2H</h1>
      <Filters />
      <SummaryCards />
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <ScoreLineChart />
        <WinLossBarChart />
      </div>
      <GamesTable />
    </div>
  );
}
