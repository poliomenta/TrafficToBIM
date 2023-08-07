import re


class Nominatim(object):
    def parse_geometry(self, geom_str):
        match = re.match(r"LINESTRING\((.+)\)", geom_str.strip())
        if match:
            return list(map(lambda x: x.split(' '), match.group(1).split(',')))
        match = re.match(r"POINT\((.+)\)", geom_str.strip())
        if match:
            return [match.group(1).split(' ')]
        return None

    def parse_street(self, street_txt: str, verbose: bool = False):
        street_txt = street_txt.replace('None', "'None'")
        match = re.match(r"\((\d+), ('|\")(.+)('|\"), '(.+)'\)", street_txt.strip())
        if match:
            osm_id = int(match.group(1))
            name = match.group(3)
            linestring_wkt = self.parse_geometry(match.group(5))
            if linestring_wkt is None or len(linestring_wkt) == 1:
                if verbose:
                    print(street_txt)
                return None
            #         linestring = wkt.loads(linestring_wkt)
            return osm_id, name, linestring_wkt
        return None


def test_parse_street():
    res = Nominatim().parse_street(
        """(554321812, 'Bolton Road', 'LINESTRING(-2.3815276 53.5418358,-2.3817707 53.5419047,-2.3818233 53.5419195,-2.3822453 53.542038,-2.3826206 53.5421584,-2.3826765 53.5421764,-2.382974 53.5422726,-2.3833405 53.5423939,-2.3849496 53.5429316,-2.3850697 53.5429756)')""")
    true_res = (554321812,
                'Bolton Road',
                [['-2.3815276', '53.5418358'],
                 ['-2.3817707', '53.5419047'],
                 ['-2.3818233', '53.5419195'],
                 ['-2.3822453', '53.542038'],
                 ['-2.3826206', '53.5421584'],
                 ['-2.3826765', '53.5421764'],
                 ['-2.382974', '53.5422726'],
                 ['-2.3833405', '53.5423939'],
                 ['-2.3849496', '53.5429316'],
                 ['-2.3850697', '53.5429756']])
    assert res == true_res

