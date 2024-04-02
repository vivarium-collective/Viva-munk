# from bigraph_schema import import_types


def REGISTER_TYPES(core):
    # core.import_types('project_we_depend_on')

    # Add a bounds type
    core.register('point2d', {
        'x': 'float',
        'y': 'float'})

    core.register('boundary', {
        'location': 'list[float]', # make this a two tuple or dict
        'diameter': 'length',
        'mass': 'mass',
        'velocity': 'length/time'})


# def REGISTER_TYPES(core):
#     core.import_types('pymunk_process')


# class TypeSystem():
#     def import_types(self, package_name):
#         register = getattr(module_lookup(f'{package_name}.REGISTER_TYPES'))
#         register(self)
