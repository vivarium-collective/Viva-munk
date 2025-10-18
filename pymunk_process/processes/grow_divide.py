from process_bigraph import Process


class GrowDivide(Process):
    config_schema = {
        'rate': 'float'}

    # def initialize(self, config):
    #     breakpoint()

    def inputs(self):
        return {
            'mass': 'float'}

    def outputs(self):
        return {
            'mass': 'float'}

    def update(self, state, interval):
        # this calculates a delta

        return {
            'mass': state['mass'] * self.config['rate'] * interval}
