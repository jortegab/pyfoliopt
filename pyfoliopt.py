import math

import numpy as np
import pandas as pd

import yfinance as yf
import pandas_datareader as web
import pandas_datareader.data as pdr

from scipy.optimize import minimize

from datetime import datetime
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import plotly.graph_objects as go

yf.pdr_override()
plt.style.use('seaborn')

___version___ = "1.0"

today = datetime.today()
START, END = today - relativedelta(years=5), today

symbols = stocks = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'IBM', 
                    'GE', 'TSLA', 'BAC', 'JPM', 'MS', 'T',
                    'C', 'BA', 'PEP', 'KO', 'YUM', 'MMM',
                    'NFLX', 'NKE', 'XOM', 'F', 'GM', 'AAL',
                    'SBUX']
futures = ['GC=F', 'CL=F', 'SI=F']
indexes = ['^GSPC', '^IXIC', '^DJI', '^N225']
bonds = ['^TYX', '^FVX', '^TNX']

#------------------------------------------------------------------------------

class GeneticAlgorithm:
    
    def __init__(self, assets):
        self._assets = assets
        self._n = len(assets)
        self._sample = Portfolio(assets)

    def _populate(self, pop_size=1000):
        
        pop = []
        for _ in range(pop_size):
            gene = np.random.random(size=self._n)
            gene /= gene.sum()
            pop.append(gene)
            
        self._initial_pop = pop
       
    def _crossover(self, p1, p2, method, **kwargs):
        
        if method == 'single point':
            cut = np.random.randint(low=1, 
                                    high=self._n)
            crossed = np.concatenate([p1[:cut], p2[cut:]], axis=None)
            return crossed / crossed.sum()
        
        elif method == 'flat':
            alpha = kwargs.get('alpha', np.random.random())
            return alpha*p1 + (1-alpha)*p2
        
    @staticmethod  
    def _mutate(pop, threshold=0.5):
        
        mutated = []
        for gene in pop:
            
            p = np.random.random()
            if p > threshold:
                mu = np.random.random()
                i, j = gene.argmax(), gene.argmin()
                if (0 <= gene[i] - mu) and (gene[j] + mu < 1):
                    gene[i] -= mu
                    gene[j] += mu
            
            mutated.append(gene)

        return mutated    

    def _fitness(self, pop):
    
        p = self._sample.mean_returns.values
        C = self._sample.covariance_matrix.values
        
        fit = []
        for gene in pop:
            expected_return = p.T @ gene
            risk = np.sqrt(gene.T @ C @ gene)
            ratio = expected_return / risk
            fit.append(ratio)
    
        return np.array(fit)

    def run(self, pop_size=1000, generations=200, elitism=True):
        
        self._populate(pop_size=pop_size)
        pop = self._initial_pop
        
        percentage = int(0.2 * pop_size)
        
        self._best_of_gen = {}
        
        for i in range(generations):
            
            new_pop = []
            
            fit = self._fitness(pop)        
            idx = fit.argsort()[::-1][:percentage]
            
            best_of_gen = fit.argmax()
            self._best_of_gen[f'Gen {i+1}'] = pop[best_of_gen]
            print(f'Best of gen {i+1}: {pop[best_of_gen]} with {fit.max()}')
            print()
            
            if elitism:
                new_pop.append(pop[best_of_gen])
                
            for _ in range(len(pop) - len(new_pop)):
                idx1, idx2 = np.random.choice(idx, size=2, replace=False)
                p1, p2 = pop[idx1], pop[idx2]
                method = np.random.choice(['single point', 'flat'])
                offspring = self._crossover(p1, p2, method=method)
                new_pop.append(offspring)
                
            new_pop = self._mutate(new_pop)
            pop = new_pop.copy() 
            
        
        best_fit = self._fitness(pop)
        best_idx = best_fit.argmax()
        best = pop[best_idx]
        
        self._best = best
        self._best_of_gen = pd.DataFrame.from_dict(self._best_of_gen, orient='index')
        self._best_of_gen.columns = self._assets


#------------------------------------------------------------------------------


class SymbolError(Exception):
    pass

class DataError(KeyError):
    pass

class FinancialError(ValueError):
    pass


class FinancialAsset:
    
    def __init__(self, symbol, db):
        
        if isinstance(symbol, str) and symbol in db:
            self._symbol = symbol
        else:
            raise SymbolError(f"'{symbol}' is not a valid symbol")
        
        try:
            self._metadata = yf.Ticker(symbol).info
        except Exception as e:
            print(f'{e}')
            print('Do you want to continue without metadata? [y/n]')
            answer = input('')    
            if answer == 'y':
                pass
            else:
                raise e
              
        try:
            self._name = self._metadata['longName']
            self._type = self._metadata['quoteType']
            self._currency = self._metadata['currency']
        except:
            pass
        
    def _get_historical_data(self, start=None, end=None, 
                             period='default', interval='1d'):
        
        if not start and period == 'default':
            start = START
            period = 'max'
        if not end and period == 'default':
            end = END
            period = 'max'
         
        data = pdr.DataReader(self._symbol, start=start, end=end,
                              period=period, interval=interval)
        
        return data
    
    def open(self, start=None, end=None):
        return self._get_historical_data(start=start, end=end)[['Open']]
    
    def high(self, start=None, end=None):
        return self._get_historical_data(start=start, end=end)[['High']]
    
    def low(self, start=None, end=None):
        return self._get_historical_data(start=start, end=end)[['Low']]
    
    def close(self, start=None, end=None):
        return self._get_historical_data(start=start, end=end)[['Close']]
    
    def adj_close(self, start=None, end=None):
        return self._get_historical_data(start=start, end=end)[['Adj Close']]
    
    def volume(self, start=None, end=None):
        return self._get_historical_data(start=start, end=end)[['Volume']]
    
    def returns(self, log=False, start=None, end=None):
        
        price = self.adj_close(start=start, end=end)
        
        if not log:
            return price.pct_change().dropna()
        else:
            rets = np.log(price) - np.log( price.shift(1) )
            return rets.dropna()
        
    @property
    def expected_return(self):
        return self.returns().mean().values[0]
    
    @property
    def risk(self):
        return self.returns().std().values[0]
    
    def plot(self, data, start=None, end=None, **kwargs):
        
        data_dict = {'open':self.open, 'high':self.high, 'low':self.low, 'close':self.close,
                     'adj_close':self.adj_close, 'volume':self.volume, 'returns':self.returns}
        
        if data != 'volume' and data in data_dict.keys():
            data = data_dict[data]
            data(start=start, end=end).plot()
            
        elif data == 'volume':
            data = data_dict['volume']
            data(start=start, end=end).plot.bar()
            
        else:
            raise DataError
            
    def candlestick(self, interval='5m'):
        
        assert interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']
        
        start = datetime(2020, 2, 28, 9, 30)
        end = datetime(2020, 2, 28, 15, 55)
        
        prices = super()._get_historical_data(start, end, interval=interval)
 
        fig = go.Figure(data=[go.Candlestick(x=prices.index,
                open=prices['Open'],
                high=prices['High'],
                low=prices['Low'],
                close=prices['Close'])])
    
        fig.update_layout(title=f'{self._ticker} Stock Price',
                          xaxis_rangeslider_visible=False)

        fig.show()
        
    """def deep_trade():
        pass"""
        
        
class Stock(FinancialAsset):
    def __init__(self, symbol):
        super().__init__(symbol, stocks)
        
class Index(FinancialAsset):
    def __init__(self, symbol):
        super().__init__(symbol, indexes)
        
class Bond(FinancialAsset):
    def __init__(self, symbol):
        super().__init__(symbol, bonds)
        
class Future(FinancialAsset):
    def __init__(self, symbol):
        super().__init__(symbol, futures)
        
        
class Portfolio:
    
    def __init__(self, assets, weights=None,
                 allow_short=True, initial_capital=1000):
        
        if all( [isinstance(asset, str) for asset in assets] ): 
            self._assets = list(sorted(assets)) 
        else:
            raise TypeError("'assets' must be a iterable of strings")
            
        self._n = len(assets)
        
        self._allow_short = allow_short 
        self._initial_capital = initial_capital
            
        if weights is None:
            self._weights = self._init_weights()
        elif self._check_weights(weights):
            self._weights = np.array([w for _,w in sorted(zip(assets,weights))])
        else:
            raise FinancialError
        
        self.returns = self._returns()
        self.mean_returns = self.returns.mean()
        self.covariance_matrix = self.returns.cov()
        self._correlations = self.returns.corr()
        
        self._risk_free = Bond('^TNX').expected_return
        self._benchmark = Index('^GSPC').expected_return
        
    def __repr__(self):
        pf = dict( zip(self._assets, self.weights) )
        return f'Portfolio({pf})'
    
    def __len__(self):
        return self._n
    
    def __getitem__(self, key):
        data = self.historical_data()
        return data.xs(key, axis=1, level=1, drop_level=False)
    
    @classmethod
    def from_dict(cls, pf, allow_short=False):
        assets = list(pf.keys())
        weights = np.array(list(pf.values()))
        return cls(assets, weights, allow_short=allow_short)
    
    """@classmethod
    def from_df(cls, df, allow_short=False):
        pass"""
    
    @classmethod
    def random(cls, n, allow_short=False):
        assets = np.random.choice(symbols, size=n, replace=False)
        assets = assets.tolist()
        return cls(assets, allow_short=allow_short)
    
    @classmethod
    def genetic(cls, assets, pop_size=1000, generations=200, elitism=True):
        
        ga = GeneticAlgorithm(assets)
        ga.run(pop_size, generations, elitism)
        
        return cls(assets, weights=ga._best)
    
    """@classmethod
    def eigen(cls, assets, n_components=2):
        pass"""
    
    def _check_weights(self, weights):
        
        if len(weights) == self._n:
            if math.isclose(sum(weights), 1):
                if not self._allow_short:
                    if all([0 <= w < 1 for w in weights]):
                        return True
                    else:
                        raise FinancialError("shorting is not allowed")
                else:
                    if all([-1 <= w <= 1 for w in weights]):
                        return True
                    else:
                        return False
            else:
                raise FinancialError("'weights' must sum up to 1")
        else:
            raise FinancialError(f"'weights' must be of length {self._n}")
        
    def _init_weights(self):
        
        if self._allow_short:
            low, high = -1.0, 1.0
        else:
            low, high = 0.0, 1.0
        
        weights = np.random.uniform(low, high, size=self._n)
        return weights / weights.sum()
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, array):
        
        if self._check_weights(array):
            self._weights = np.array(array)
        else:
            raise FinancialError
            
    def reset_weights(self):
        self.weights = self._init_weights()  
        
    def set_weights(self, weights:dict): 
        weights = dict(sorted(weights.items()))
        self.weights = list(weights.values())
            
    def _get_historical_data(self, start=None, end=None):
        
        if not start:
            start = START
        if not end:
            end = END
        
        data = pdr.DataReader(self._assets, start=start, end=end, 
                              group_by='ticker')
        
        data.columns = pd.MultiIndex.from_tuples((sorted(data.columns, 
                                                         key=lambda x: x[0])))
        data.drop(data.tail(1).index, inplace=True)
        return data        
    
    def historical_data(self):
        return self._get_historical_data()
    
    def _returns(self, log=False, start=None, end=None):
        
        price = self._get_historical_data(start=start, end=end)
        price = price.xs('Adj Close', axis=1, level=1)
        price = pd.DataFrame(price)
        
        if not log:
            return price.pct_change().dropna()
        else:
            rets = np.log(price) - np.log( price.shift(1) )
            return rets.dropna()
    
    def portfolio_returns(self):
        df = self.weights @ self.returns.T
        df = pd.DataFrame(df)
        df.columns = ['Portfolio Returns']
        return df
    
    @property   
    def expected_return(self):
        return self.mean_returns.values.T @ self.weights    
    
    @property
    def risk(self):
        sigma = self.weights.T @ self.covariance_matrix.values @ self.weights
        return np.sqrt(sigma)
    
    @property
    def sharpe_ratio(self):
        return (self.expected_return - self._risk_free) / self.risk
    
    """@property
    def information_ratio(self):
        return (self.expected_return - self._benchmark) / self.risk"""
    
    """@property
    def max_drawdown(self):
        pass
    
    @property
    def calmar_ratio(self):
        pass"""
    
#------------------------------------------------------------------------------
    
    def performance(self):
        
        benchmark = web.DataReader('^GSPC', data_source='yahoo', 
                                   start=START, end=END)[['Adj Close']]
        
        capital = self.portfolio_returns() + 1
        capital = capital.cumprod() * benchmark.values[0]
        
        ax = plt.subplot(title='Portfolio Performance vs S&P 500 Benchmark')
        ax.plot(benchmark)
        ax.plot(capital)
        ax.legend(['Benchmark', 'Portfolio'])
        plt.show()
                
    def _ret(self, w):
        
        p = self.mean_returns.values
        expected_return = p.T @ w
        
        return expected_return
    
    def _std(self, w):
        C = self.covariance_matrix.values
        return np.sqrt(w.T @ C @ w)
    
    def _neg_sharpe_ratio(self, w):
        
        p = self.mean_returns.values
        C = self.covariance_matrix.values
        
        expected_return = p.T @ w
        risk = np.sqrt(w.T @ C @ w)
        
        return - (expected_return - self._risk_free) / risk
    
    def _neg_info_ratio(self, w):

        raise NotImplementedError
        
        """p = self.mean_returns.values
        C = self.covariance_matrix.values
        
        expected_return = p.T @ w
        risk = np.sqrt(w.T @ C @ w)
        
        return - (expected_return - self._benchmark) / risk"""
    
    def efficient_frontier(self, show=True):
        
        max_mu = - self.optimize(target='return', allocate=False).fun
        
        w = []
        for mu in np.linspace(0, max_mu, 100):
            fun = self._std
            w0 = self.weights
            bounds = [(-1,1) for _ in range(self._n)] if self._allow_short else [(0,1) for _ in range(self._n)]
            const = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
                     {'type':'eq', 'fun': lambda x: self._ret(x) - mu})
            
            results = minimize(fun, x0=w0, method='SLSQP', bounds=bounds,
                          constraints=const)
            
            w.append(results.x)
         
        std = [self._std(pf) for pf in w] 
        rets = [self._ret(pf) for pf in w]
        
        ef = list(zip(std, rets))
        ef = sorted(ef, key=lambda x: x[-1])
        
        ef_x = [pf[0] for pf in ef]
        ef_y = [pf[1] for pf in ef]
    
        if show:
            plt.plot(ef_x, ef_y)
            plt.scatter(self._std(self.weights), self._ret(self.weights), c='g')
            plt.show()
        
        return ef_x, ef_y
    
    def efficient_return(self, target_risk, allocate=False):
        
        fun = lambda x: -self._ret(x)
        w0 = self.weights
        bounds = [(-1,1) for _ in range(self._n)] if self._allow_short else [(0,1) for _ in range(self._n)]
        const = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
                 {'type':'eq', 'fun': lambda x: self._std(x) - target_risk})
            
        results = minimize(fun, x0=w0, method='SLSQP', bounds=bounds,
                           constraints=const)
        
        if allocate:
            self.weights = results.x
            
        return results
    
    def efficient_risk(self, target_return, allocate=False):
        
        fun = self._std
        w0 = self.weights
        bounds = [(-1,1) for _ in range(self._n)] if self._allow_short else [(0,1) for _ in range(self._n)]
        const = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
                 {'type':'eq', 'fun': lambda x: self._ret(x) - target_return})
            
        results = minimize(fun, x0=w0, method='SLSQP', bounds=bounds,
                           constraints=const)
        
        if allocate:
            self.weights = results.x
            
        return results

    def optimize(self, target='sharpe', allocate=True):
        
        w0 = self.weights
        bounds = [(-1,1) for _ in range(self._n)] if self._allow_short else [(0,1) for _ in range(self._n)]
        const = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
        
        if target == 'sharpe':
            
            fun = self._neg_sharpe_ratio            
            results = minimize(fun, x0=w0, method='SLSQP', bounds=bounds,
                               constraints=const)
        
        elif target == 'return':
            
            fun = lambda x: - self._ret(x)
            results = minimize(fun, x0=w0, method='SLSQP', bounds=bounds,
                              constraints=const)
        
        elif target == 'risk':
            
            fun = self._std
            results = minimize(fun, x0=w0, method='SLSQP', bounds=bounds,
                              constraints=const)
            
        elif target == 'info':
            
            fun = self._neg_info_ratio            
            results = minimize(fun, x0=w0, method='SLSQP', bounds=bounds,
                               constraints=const)
            
        else:
            raise FinancialError
            
        if allocate:
            self.weights = results.x
            
        return results
        
    def tearsheet(self):
        
        capital = self.portfolio_returns() + 1
        capital = capital.cumprod() * self._initial_capital
        
        ef_x, ef_y = self.efficient_frontier(show=False)
        
        
#-------Plot portfolio weight distribution-------------------------------------
        ax1 = plt.subplot(212, title='Portfolio Weights')
        
        if self._allow_short:
            
            ax1.bar(range(self._n), self.weights, tick_label=self._assets, 
                    color=(pd.Series(self.weights) > 0).map({True:'g', False:'r'}))
            
            colors = {'Long Position':'g', 'Short Position':'r'}
            labels = list(colors.keys())
            handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
            ax1.legend(handles, labels)
            
        else:
            ax1.bar(range(self._n), self.weights, tick_label=self._assets)
       
        
#-------Plot portfolio cumulative capital--------------------------------------        
        ax2 = plt.subplot(221, title='Portfolio Performance')
        ax2.plot(capital, c='g') 
        ax2.legend(['Portfolio Cumulative Capital'])
        

#-------Plot portfolio location with respect to efficient frontier-------------
        ax3 = plt.subplot(222, title='Portfolio Efficient Frontier')
        ax3.plot(ef_x, ef_y)
        ax3.scatter(self._std(self.weights), self._ret(self.weights), c='g')
        ax3.legend(['Efficient Portfolios', 'Actual Portfolio'])
       
        plt.show()
        
        
class TearSheet:
    def __init__(self, pf):
        pass

class Strategy:
    def __init__(self, id, allow_short=False, freq='daily'):
        pass
    
class Backtest:
    def __init__(self, assets, strategy_id, benchmark='^GSPC', 
                 initial_capital=10000, start=None, end=None, period='default'):
        pass
    def history() -> pd.DataFrame:
        pass
    def summary():
        pass