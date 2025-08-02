(define (problem prob)
  (:domain pr2-blocksworld)
  (:objects
    red green blue yellow grey
  )
  (:init
    (arm-empty)
    (on-table blue)
    (on green blue)
    (on red green)
    (on-table grey)
    (on yellow grey)
    (clear red)
    (clear yellow)
  )
  (:goal
    (and
    (on-table red)
    (on green red)
    (on blue green)
    (on-table yellow)
    (on grey yellow)
     )))
