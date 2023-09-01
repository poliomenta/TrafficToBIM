from .gmal import GMAL
from .od_matrix import AreaOriginDestinationMatrix


class ODMatrixAnalysis(object):
    def __init__(self, *args, **kwargs):
        self.area_od_matrix = AreaOriginDestinationMatrix(*args, **kwargs)
        self.gmal = GMAL(*args, **kwargs)
        self.gmal.load()

    def compare_population_and_od_commuter_numbers(self, attribute: str = 'all', need_plot: bool = True):
        from scipy.stats import kendalltau
        import matplotlib.pyplot as plt

        od_from_all_sum = self.area_od_matrix.get_grouped_geocode_sum(attribute)
        od_from_all_sum['area_code'] = od_from_all_sum['geo_code1']
        od_from_all_sum_population = \
            od_from_all_sum.merge(self.gmal.df[['area_code', 'all_residents_2011']], on='area_code')
        kendall_score = kendalltau(od_from_all_sum_population[attribute], od_from_all_sum_population['all_residents_2011'])
        print(f'{kendall_score=}')
        if need_plot:
            plt.scatter(od_from_all_sum_population[attribute], od_from_all_sum_population['all_residents_2011'])
            plt.xlabel(f'Commuters: {attribute}')
            plt.ylabel('all_residents_2011')
            plt.show()
        return kendall_score
