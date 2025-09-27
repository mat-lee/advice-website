import React, { useState, useMemo, useEffect } from 'react';
import { Search, ChevronUp, ChevronDown, RotateCcw, AlertCircle, Loader, Copy } from 'lucide-react';
import Papa from 'papaparse';

function AdviceAggregator() {
  const [adviceData, setAdviceData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedGroup, setSelectedGroup] = useState('');
  const [selectedSubreddit, setSelectedSubreddit] = useState('');
  const [sortBy, setSortBy] = useState('quality_score');
  const [sortOrder, setSortOrder] = useState('desc');
  const [hoveredItem, setHoveredItem] = useState(null);
  const [defaultGroupSet, setDefaultGroupSet] = useState(false);

  // normalize helper
  const norm = (s) => (typeof s === 'string' ? s.trim().toLowerCase() : '');

  // Robust key generation
  const mkKey = (row, i) => {
    const s = [
      row.id ?? '',
      row.subreddit ?? '',
      row.category ?? '',
      row.advice ?? '',
      i
    ].join('|');
    let h = 5381;
    for (let j = 0; j < s.length; j++) h = (h * 33) ^ s.charCodeAt(j);
    return 'k_' + (h >>> 0).toString(36);
  };

  useEffect(() => {
    const loadCSVData = async () => {
      try {
        setLoading(true);
        const response = await fetch('/advice-website/display_data.csv');
        if (!response.ok) throw new Error(`Failed to load CSV file: ${response.statusText}`);
        const csvText = await response.text();

        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
          transform: (value, field) => {
            if (field === 'upvotes') return parseInt(value) || 0;
            if (field === 'upvote_ratio') return parseFloat(value) || 0;
            if (field === 'quality_score') return parseFloat(value) || 0;
            if (typeof value === 'string') return value.trim();
            return value;
          },
          complete: (results) => {
            const validData = results.data.filter(
              (row) => row && typeof row.advice === 'string' && row.advice.trim().length > 0
            );
            const normalized = validData.map((row, i) => ({
              ...row,
              category_norm: norm(row.category),
              subreddit_norm: norm(row.subreddit),
              advice_norm: norm(row.advice),
              __key: mkKey(row, i),
            }));            
            setAdviceData(normalized);
            setLoading(false);
          },
          error: (error) => {
            setError(`CSV parsing error: ${error.message}`);
            setLoading(false);
          },
        });
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    loadCSVData();
  }, []);

  // options for selects
  const groupOptions = useMemo(() => {
    const byNorm = new Map();
    for (const item of adviceData) {
      if (!item.category_norm) continue;
      if (!byNorm.has(item.category_norm)) byNorm.set(item.category_norm, item.category?.trim() || 'Uncategorized');
    }
    return Array.from(byNorm.entries()).map(([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label));
  }, [adviceData]);

  // Set default group when data loads
  useEffect(() => {
    if (adviceData.length > 0 && !defaultGroupSet && groupOptions.length > 0) {
      setSelectedGroup(groupOptions[0].value);
      setDefaultGroupSet(true);
    }
  }, [adviceData, groupOptions, defaultGroupSet]);

  const subredditOptions = useMemo(() => {
    const byNorm = new Map();
    for (const item of adviceData) {
      if (!item.subreddit_norm) continue;
      if (!byNorm.has(item.subreddit_norm)) byNorm.set(item.subreddit_norm, item.subreddit?.trim() || 'Unknown');
    }
    return Array.from(byNorm.entries()).map(([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label));
  }, [adviceData]);

  // filtering - require a group to be selected
  const filteredData = useMemo(() => {
    // Don't show anything if no group is selected
    if (!selectedGroup) return [];

    let data = adviceData;

    const s = norm(searchTerm);
    if (s) {
      data = data.filter(
        (item) =>
          item.advice_norm.includes(s) ||
          item.category_norm.includes(s) ||
          item.subreddit_norm.includes(s)
      );
    }

    if (selectedGroup) {
      data = data.filter((item) => item.category_norm === selectedGroup);
    }

    if (selectedSubreddit) {
      data = data.filter((item) => item.subreddit_norm === selectedSubreddit);
    }

    // sort
    const arr = [...data];
    arr.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];

      if (aValue == null) aValue = sortBy === 'upvotes' ? 0 : sortBy === 'upvote_ratio' ? 0 : '';
      if (bValue == null) bValue = sortBy === 'upvotes' ? 0 : sortBy === 'upvote_ratio' ? 0 : '';

      if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }
      if (sortOrder === 'asc') return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
    });

    return arr;
  }, [adviceData, searchTerm, selectedGroup, selectedSubreddit, sortBy, sortOrder]);

  const visibleData = filteredData;

  const clearFilters = () => {
    setSearchTerm('');
    setSelectedSubreddit('');
    // Don't clear selectedGroup - keep it selected
  };

  const formatNumber = (num) => (num >= 1000 ? (num / 1000).toFixed(1) + 'k' : num.toString());
  
  const copyToClipboard = (text, e) => { 
    e.stopPropagation(); 
    navigator.clipboard?.writeText(text);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading advice...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center max-w-md p-8">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-medium text-gray-900 mb-2">Unable to load data</h2>
          <p className="text-gray-600 text-sm mb-6">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-black text-white rounded hover:bg-gray-800 transition-colors text-sm"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Simple Header */}
      <div className="border-b border-gray-200 bg-white sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-medium text-gray-900 mb-4">Advice</h1>
          
          {/* Search and Filters - Single Row */}
          <div className="flex flex-wrap gap-3 items-center">
            <div className="relative flex-1 min-w-64">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search advice..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
              />
            </div>

            <select
              value={selectedGroup}
              onChange={(e) => setSelectedGroup(e.target.value || '')}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
            >
              <option value="">Select a Category</option>
              {groupOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>

            <select
              value={selectedSubreddit}
              onChange={(e) => setSelectedSubreddit(e.target.value || '')}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
            >
              <option value="">All Sources</option>
              {subredditOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>

            <div className="flex items-center">
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-l-md text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
              >
                <option value="upvotes">Popularity</option>
                <option value="upvote_ratio">Quality</option>
                <option value="quality_score">AI Quality</option>
                <option value="category">Category</option>
              </select>
              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="px-3 py-2 border border-l-0 border-gray-300 rounded-r-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-900"
                title={`Sort ${sortOrder === 'asc' ? 'Descending' : 'Ascending'}`}
              >
                {sortOrder === 'asc' ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
            </div>

            {(searchTerm || selectedSubreddit) && (
              <button
                onClick={clearFilters}
                className="flex items-center px-3 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Clear
              </button>
            )}
          </div>

          {/* Results Count */}
          <div className="mt-3 text-sm text-gray-500">
            {selectedGroup ? (
              <>
                {visibleData.length} {visibleData.length === 1 ? 'result' : 'results'}
              </>
            ) : (
              'Select a category to view advice'
            )}
          </div>
        </div>
      </div>

      {/* Advice List */}
      <div className="max-w-4xl mx-auto px-6 py-3">
        <div className="divide-y divide-gray-100">
          {visibleData.map((item) => (
            <div
              key={item.__key}
              className="group relative py-2 px-4 hover:bg-gray-50 rounded-lg transition-colors cursor-pointer"
              onMouseEnter={() => setHoveredItem(item)}
              onMouseLeave={() => setHoveredItem(null)}
            >
              {/* Advice Text */}
              <div className="text-gray-900 leading-relaxed mb-2">
                {item.advice}
              </div>

              {/* Metadata Row - completely minimal, no category shown */}
              <div className="flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center">
                  {/* Empty - no metadata shown by default */}
                </div>
                
                <div className="flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => copyToClipboard(item.advice, e)}
                    className="p-1 hover:text-gray-900 transition-colors"
                    title="Copy advice"
                  >
                    <Copy className="w-3 h-3" />
                  </button>
                </div>
              </div>

              {/* Hover Tooltip with detailed info */}
              {hoveredItem === item && (
                <div className="absolute right-4 top-2 bg-black text-white text-xs px-3 py-2 rounded shadow-lg z-20 min-w-48">
                  <div className="space-y-1">
                    <div><strong>ID:</strong> {item.id || 'N/A'}</div>
                    <div><strong>Source:</strong> {item.subreddit || 'Unknown'}</div>
                    <div><strong>Upvotes:</strong> {formatNumber(item.upvotes)}</div>
                    <div><strong>Quality:</strong> {Math.round((item.upvote_ratio || 0) * 100)}% positive</div>
                    <div><strong>AI Quality:</strong> {((item.quality_score || 0) * 100).toFixed(0)}%</div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* No Results */}
        {visibleData.length === 0 && selectedGroup && (
          <div className="text-center py-12">
            <p className="text-gray-500">No advice found in this category matching your criteria.</p>
          </div>
        )}

        {/* No Category Selected */}
        {!selectedGroup && (
          <div className="text-center py-12">
            <p className="text-gray-500">Please select a category to view advice.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default AdviceAggregator;