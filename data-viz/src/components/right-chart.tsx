import { FC, useEffect, useMemo, useRef, useState } from "react";
import "echarts-gl";
import * as echarts from "echarts";
import axios from "axios";
import { EChartsOption } from "echarts";

export const RightChart: FC<{ iter: number }> = ({ iter }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const client = useMemo(
    () => axios.create({ baseURL: "http://localhost:4000" }),
    []
  );

  const [chart, setchart] = useState<echarts.ECharts>();
  const min = 0.95;
  const max = 1;

  const getOption = (
    x: number[],
    y: number[],
    z: number[],
    c: number[],
    m_x: number[],
    m_y: number[],
    m_z: number[]
  ): EChartsOption => ({
    title: {
      text: "Mean Estimate",
      left: "40%",
    },
    tooltip: {},
    visualMap: [
      {
        top: 10,
        calculable: true,
        dimension: 3,
        max: max,
        min: min,
        precision: 4,
        seriesIndex: 0,
        inRange: {
          symbolSize: [5, 15],
          color: [
            "#efdbff",
            "#adc6ff",
            "#91d5ff",
            "#5cdbd3",
            "#95de64",
            "#bae637",
            "#ffec3d",
            "#faad14",
            "#fa8c16",
            "#d4380d",
            "#a8071a",
            "#780650",
          ],
          colorAlpha: [0.5, 1],
        },
        textStyle: {
          color: "#000",
        },
      },
    ],
    xAxis3D: {
      name: "Sigmoidal Kernel",
      type: "value",
      nameGap: 5,
      min: Math.min(...x),
      max: Math.max(...x),
      nameTextStyle: {
        fontSize: 12,
      },
    },
    yAxis3D: {
      name: "Gaussian Kernel",
      type: "value",
      nameGap: 5,
      min: Math.min(...y),
      max: Math.max(...y),
      nameTextStyle: {
        fontSize: 12,
      },
    },
    zAxis3D: {
      name: "Mixing Ratio",
      type: "value",
      nameGap: 5,
      nameTextStyle: {
        fontSize: 12,
      },
    },
    grid3D: {
      axisLine: {
        lineStyle: {
          color: "#000",
        },
      },
      axisPointer: {
        show: false,
        lineStyle: {
          color: "#ffbd67",
        },
      },
    },
    series: [
      {
        type: "scatter3D",
        data: x.map((item, i) => [x[i], y[i], z[i], c[i]]),
        emphasis: {
          itemStyle: {
            borderWidth: 2,
            borderColor: "#fff",
          },
        },
      } as any,
      {
        type: "scatter3D",
        data: m_x.map((item, i) => [m_x[i], m_y[i], m_z[i]]),
        symbol: "pin",
        symbolSize: 20,
        itemStyle: {
          opacity: 1,
          color: "#096dd9",
        },
        // label: {
        //   show: true,
        //   position: 'top',
        //   distance: 0,
        //   formatter: (p: any) => p.dataIndex + 1,
        //   textStyle: {
        //     color: '#000'
        //   }
        // },
        emphasis: {
          itemStyle: {
            borderWidth: 2,
            borderColor: "#fff",
          },
        },
      } as any,
      // {
      //   type: 'line3D',
      //   data: m_x.map((_, i) => [m_x[i], m_y[i], m_z[i]]),
      //   lineStyle: {
      //     color: '#000',
      //     width: 5
      //   },
      //   zlevel: 100
      // },
      {
        type: "line3D",
        data: [
          [m_x.slice(-1)[0], m_y.slice(-1)[0], 0],
          [m_x.slice(-1)[0], m_y.slice(-1)[0], 1],
        ],
        lineStyle: {
          color: "#096dd9",
          width: 5,
        },
        zlevel: 100,
      },
      {
        type: "line3D",
        data: [
          [-6, m_y.slice(-1)[0], m_z.slice(-1)[0]],
          [-10, m_y.slice(-1)[0], m_z.slice(-1)[0]],
        ],
        lineStyle: {
          color: "#096dd9",
          width: 5,
        },
        zlevel: 100,
      },
      {
        type: "line3D",
        data: [
          [m_x.slice(-1)[0], -4, m_z.slice(-1)[0]],
          [m_x.slice(-1)[0], -8, m_z.slice(-1)[0]],
        ],
        lineStyle: {
          color: "#096dd9",
          width: 5,
        },
        zlevel: 100,
      },
    ],
  });

  useEffect(() => {
    client.get("/", { params: { iter } }).then((res) => {
      client.get("/mark", { params: { iter } }).then((res2) => {
        if (chartRef.current) {
          if (!chart) {
            const c = echarts.init(chartRef.current);
            setchart(c);
            c.setOption(
              getOption(
                res.data.x,
                res.data.y,
                res.data.z,
                res.data.c,
                res2.data.x,
                res2.data.y,
                res2.data.z
              ),
              true
            );
          } else {
            chart.setOption(
              getOption(
                res.data.x,
                res.data.y,
                res.data.z,
                res.data.c,
                res2.data.x,
                res2.data.y,
                res2.data.z
              ),
              true
            );
          }
        }
      });
    });
  }, [chart, client, iter]);

  return <div style={{ width: "50vw", height: "90vh" }} ref={chartRef} />;
};
