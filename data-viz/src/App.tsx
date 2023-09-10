import { Col, Row, Slider } from "antd";
import { FC, useState } from "react";
import { LeftChart } from "./components/left-chart";
import { RightChart } from "./components/right-chart";

export const App: FC = () => {
  const [iter, setiter] = useState(1);

  return (
    <>
      <Row style={{ width: "100vw", paddingLeft: 50, paddingRight: 50 }}>
        <Slider
          style={{ width: "100%" }}
          min={1}
          max={24}
          onChange={(val: number) => setiter(val)}
          value={iter}
        />
      </Row>
      <Row justify='center'>
        <Col span={12} style={{ height: "max-large" }}>
          <LeftChart iter={iter} />
        </Col>
        <Col span={12} style={{ height: "max-large" }}>
          <RightChart iter={iter} />
        </Col>
      </Row>
    </>
  );
};
