package indexparamcheck

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/pkg/util/metric"
)

func Test_cagraChecker_CheckTrain(t *testing.T) {
	p1 := map[string]string{
		DIM:    strconv.Itoa(128),
		Metric: metric.L2,
	}
	p2 := map[string]string{
		DIM:    strconv.Itoa(128),
		Metric: metric.IP,
	}
	p3 := map[string]string{
		DIM:                strconv.Itoa(128),
		Metric:             metric.L2,
		CAGRA_INTER_DEGREE: 20,
	}

	p4 := map[string]string{
		DIM:                strconv.Itoa(128),
		Metric:             metric.L2,
		CAGRA_GRAPH_DEGREE: 20,
	}
	p5 := map[string]string{
		DIM:                strconv.Itoa(128),
		Metric:             metric.L2,
		CAGRA_INTER_DEGREE: 60,
		CAGRA_GRAPH_DEGREE: 20,
	}
	p6 := map[string]string{
		DIM:                strconv.Itoa(128),
		Metric:             metric.L2,
		CAGRA_INTER_DEGREE: 20,
		CAGRA_GRAPH_DEGREE: 60,
	}
	p7 := map[string]string{
		DIM:    strconv.Itoa(128),
		Metric: metric.SUPERSTRUCTURE,
	}

	cases := []struct {
		params   map[string]string
		errIsNil bool
	}{
		{p1, true},
		{p2, false},
		{p3, true},
		{p4, true},
		{p5, true},
		{p6, false},
		{p7, false},
	}

	c := newCagraChecker()
	for _, test := range cases {
		err := c.CheckTrain(test.params)
		if test.errIsNil {
			assert.NoError(t, err)
		} else {
			assert.Error(t, err)
		}
	}
}
